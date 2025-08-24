from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, Dict, Any

from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_default_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def build_mutual_knn(Z: np.ndarray, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于 latent Z (n x d) 构建 mutual kNN 图。
    返回 rows, cols（均为 E 长度）
    """
    n = Z.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(Z)
    dist, idx = nn.kneighbors(Z)
    idx = idx[:, 1:]   # 去掉自己
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)

    # 互为邻居筛选
    E = set(zip(rows.tolist(), cols.tolist()))
    Emut = [(i, j) for (i, j) in E if (j, i) in E]
    rows = np.array([i for (i, _) in Emut], dtype=np.int32)
    cols = np.array([j for (_, j) in Emut], dtype=np.int32)
    return rows, cols

def edge_features(
    Z: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    batches: Optional[np.ndarray] = None,
    delta_mu: Optional[np.ndarray] = None,
    celltypes: Optional[np.ndarray] = None,
    include_density: bool = True,
    density_k: int = 15
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    生成边特征 Phi_ij。
    维度（按列拼接）：[||dz||, ||dz||^2, <dz, Δμ>, dens_diff, same_batch, cross_batch, same_type]
    缺省项会置 0。
    """
    dz = Z[rows] - Z[cols]
    d = np.linalg.norm(dz, axis=1)
    d2 = d**2

    if delta_mu is None:
        delta_mu = np.zeros(Z.shape[1], dtype=Z.dtype)
    proj = (dz * delta_mu[None, :]).sum(axis=1)

    # 简易密度：每点的均邻距（knn），dens_diff=ρ_i-ρ_j
    if include_density:
        nn = NearestNeighbors(n_neighbors=density_k + 1).fit(Z)
        dist, _ = nn.kneighbors(Z)
        rho = dist[:, 1:].mean(axis=1)  # 每个点的平均邻距
        dens_diff = rho[rows] - rho[cols]
    else:
        dens_diff = np.zeros_like(d)

    if batches is not None:
        same_batch = (batches[rows] == batches[cols]).astype(float)
        cross_batch = 1.0 - same_batch
    else:
        same_batch = np.zeros_like(d)
        cross_batch = np.zeros_like(d)

    if celltypes is not None:
        same_type = (celltypes[rows] == celltypes[cols]).astype(float)
    else:
        same_type = np.zeros_like(d)

    Phi = np.stack([d, d2, proj, dens_diff, same_batch, cross_batch, same_type], axis=1)
    meta = dict(delta_mu=delta_mu, density_k=density_k)
    return Phi.astype(np.float32), meta

def gaussian_reference_weights(
    Z: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    sigma: Optional[float] = None
) -> np.ndarray:
    """
    固定高斯核边权 w0_ij（仅用于训练对齐的软约束，非必须）。
    """
    d2 = np.sum((Z[rows] - Z[cols])**2, axis=1)
    if sigma is None:
        sigma = float(np.sqrt(np.median(d2) + 1e-8))
    w0 = np.exp(-d2 / (2 * sigma * sigma))
    return w0.astype(np.float32)


class EdgeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.1, learn_temp: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )
        self.learn_temp = learn_temp
        self.temp = nn.Parameter(torch.tensor(0.5)) if learn_temp else None

    def forward(self, phi_ij: torch.Tensor) -> torch.Tensor:
        """
        phi_ij: (E, F)
        return: scores (E,) —— 未做行softmax的边打分/温度归一
        """
        s = self.net(phi_ij).squeeze(-1)
        if self.learn_temp:
            tau = torch.clamp(self.temp, 1e-2, 10.0)
            s = s / tau
        return s

def row_softmax_to_kernel(
    rows: np.ndarray,
    cols: np.ndarray,
    scores: torch.Tensor,
    n_nodes: int,
    add_self_loop: bool = True
) -> sp.csr_matrix:
    """
    每个源节点 i 的出边 scores[i->j] 做行 softmax，得到核 K（稀疏行归一）。
    这里在 device 上做 softmax，最后转回 CPU 的 scipy CSR。
    """
    device = scores.device
    scores = scores.float()  # 强制 FP32
    rows_t = torch.as_tensor(rows, dtype=torch.long, device=device)
    cols_t = torch.as_tensor(cols, dtype=torch.long, device=device)

    n = n_nodes
    max_row = torch.full((n,), torch.finfo(torch.float32).min, device=device, dtype=torch.float32)
    max_row = torch.scatter_reduce(max_row, 0, rows_t, scores, reduce="amax", include_self=False)

    ex = torch.exp(scores - max_row[rows_t])
    denom = torch.zeros((n,), device=device, dtype=torch.float32).scatter_add_(0, rows_t, ex)
    val = ex / (denom[rows_t] + 1e-12)

    K = sp.coo_matrix((_to_numpy(val), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    if add_self_loop:
        K = K + sp.eye(n_nodes, format="csr")
        Dinv = sp.diags(1.0 / np.maximum(np.array(K.sum(1)).ravel(), 1e-8))
        K = Dinv @ K
    return K


def train_edge_mlp_one_epoch(
    model: EdgeMLP,
    Phi: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    Z: np.ndarray,
    *,
    w0: Optional[np.ndarray] = None,
    proj: Optional[np.ndarray] = None,
    lr: float = 1e-3,
    weight_align: float = 1.0,
    weight_margin: float = 0.2,
    weight_lap: float = 0.05,
    device: Optional[str] = None,
) -> float:
    """
    自监督损失（全 FP32）：
      L = wa * 对齐固定核 + wm * 边界方向 margin + wl * 拉普拉斯正则
    """
    dev = get_default_device(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # 全部 float32 + 放到 device
    phi_t  = torch.as_tensor(Phi,  dtype=torch.float32, device=dev)
    rows_t = torch.as_tensor(rows, dtype=torch.long,    device=dev)
    cols_t = torch.as_tensor(cols, dtype=torch.long,    device=dev)

    opt.zero_grad(set_to_none=True)

    scores = model(phi_t).float()  # (E,) 确保 FP32
    n = Z.shape[0]

    # (1) 对齐固定核（可选）
    L_align = torch.tensor(0.0, device=dev, dtype=torch.float32)
    if w0 is not None:
        w0_t = torch.as_tensor(w0, dtype=torch.float32, device=dev)
        denom = torch.zeros((n,), dtype=torch.float32, device=dev).scatter_add_(0, rows_t, w0_t)
        w0_row = w0_t / (denom[rows_t] + 1e-12)

        # 行 softmax（数值稳定）
        max_row = torch.full((n,), torch.finfo(torch.float32).min, device=dev, dtype=torch.float32)
        max_row = torch.scatter_reduce(max_row, 0, rows_t, scores, reduce="amax", include_self=False)
        ex = torch.exp(scores - max_row[rows_t])
        denom_s = torch.zeros((n,), dtype=torch.float32, device=dev).scatter_add_(0, rows_t, ex)
        s_row = ex / (denom_s[rows_t] + 1e-12)

        L_align = F.mse_loss(s_row, w0_row)

    # (2) 边界方向 margin（鼓励与 Δμ 同向的边得分更高）
    L_margin = torch.tensor(0.0, device=dev, dtype=torch.float32)
    if proj is not None:
        proj_t = torch.as_tensor(proj, dtype=torch.float32, device=dev)
        margin = 0.5
        L_margin = torch.relu(margin - torch.sign(proj_t) * scores).mean()

    # (3) 拉普拉斯正则（防奇异注意力）
    L_lap = torch.tensor(0.0, device=dev, dtype=torch.float32)
    if Z is not None:
        Zi = torch.as_tensor(Z[rows], dtype=torch.float32, device=dev)
        Zj = torch.as_tensor(Z[cols], dtype=torch.float32, device=dev)
        dist2 = ((Zi - Zj) ** 2).sum(1)
        L_lap = (torch.exp(scores) * dist2).mean()

    loss = weight_align * L_align + weight_margin * L_margin + weight_lap * L_lap
    loss.backward()
    opt.step()
    return float(loss.item())

def brp_observed(K: sp.csr_matrix, y: np.ndarray, lam: float = 0.9, T: int = 30, prior: Optional[float] = None, eps: float = 1e-6):
    y = y.astype(float)
    p_q = y.copy()
    p_r = 1.0 - y
    for _ in range(T):
        p_q = (1 - lam) * y       + lam * K.dot(p_q)
        p_r = (1 - lam) * (1 - y) + lam * K.dot(p_r)
    if prior is None:
        prior = float(y.mean())
    logit = np.log(p_q + eps) - np.log(p_r + eps) - np.log(prior + eps) + np.log(1 - prior + eps)
    p = 1.0 / (1.0 + np.exp(-logit))
    return logit, p

def brp_permutation_pvals(K: sp.csr_matrix, y: np.ndarray, R: int = 800, lam: float = 0.9, T: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    logit_obs, p_obs = brp_observed(K, y, lam=lam, T=T, prior=y.mean())
    ge = np.zeros_like(y, dtype=np.int32)
    for _ in range(R):
        y_perm = rng.permutation(y)
        logit_perm, _ = brp_observed(K, y_perm, lam=lam, T=T, prior=y.mean())
        ge += (logit_perm >= logit_obs).astype(np.int32)  # right-tail test
    pval = (1 + ge) / (1 + R)
    return logit_obs, p_obs, pval

def benjamini_hochberg(pvals: np.ndarray):
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q); out[order] = q
    return np.clip(out, 0, 1)

# =========================

def local_oor_detector(
    Z: np.ndarray,
    y: np.ndarray,                  # 0=ref/atlas, 1=query
    batches: np.ndarray,            # 批次标签（置换时用）
    *,
    k: int = 20,
    celltypes: Optional[np.ndarray] = None,   # 可选（边特征）
    tail_direction_mu: Optional[np.ndarray] = None,  # Δμ（若 None 自动用 y 计算）
    train_epochs: int = 50,
    lr: float = 1e-3,
    weight_align: float = 1.0,
    weight_margin: float = 0.2,
    weight_lap: float = 0.05,
    lam: float = 0.9,
    T: int = 30,
    R_perm: int = 800,
    alpha_fdr: float = 0.10,
    seed: int = 42,
    device: Optional[str] = None,   # 'cuda' / 'cpu'
) -> Dict[str, Any]:

    set_seed(seed)
    dev = get_default_device(device)

    # mnn
    rows, cols = build_mutual_knn(Z, k=k)

    # Δμ
    if tail_direction_mu is None:
        mu_q = Z[y == 1].mean(axis=0)
        mu_r = Z[y == 0].mean(axis=0)
        delta_mu = mu_q - mu_r
    else:
        delta_mu = tail_direction_mu

    # edge feature
    print("- Constructing edge features...")
    Phi, ef_meta = edge_features(Z, rows, cols, batches=batches, delta_mu=delta_mu, celltypes=celltypes)

    # reference gaussian kernel (optional)
    w0 = gaussian_reference_weights(Z, rows, cols, sigma=None)

    # train Edge-MLP (FP32)
    print(f"- Training kernel on {dev} (fp32) ...")
    model = EdgeMLP(in_dim=Phi.shape[1], hidden=64, dropout=0.1, learn_temp=True).to(dev).float()

    loop = tqdm(range(train_epochs), total=train_epochs, desc="Epochs")
    for ep in loop:
        loss = train_edge_mlp_one_epoch(
            model, Phi, rows, cols, Z,
            w0=w0, proj=Phi[:, 2],  # 第3列是 <dz, Δμ>
            lr=lr, weight_align=weight_align, weight_margin=weight_margin, weight_lap=weight_lap,
            device=dev.type
        )
        loop.set_postfix(kernel_converge_loss=loss)

    with torch.no_grad():
        scores = model(torch.as_tensor(Phi, dtype=torch.float32, device=dev)).float()
    K = row_softmax_to_kernel(rows, cols, scores, n_nodes=Z.shape[0], add_self_loop=True)

    # (7) BRP observed + global permutation p-values（CPU）
    print("- Running BRP on the graph...")
    logit, prob, pval = brp_permutation_pvals(K, y, R=R_perm, lam=lam, T=T, seed=seed)

    # (8) BH FDR control
    print("- FDR control with significance value ",alpha_fdr,"...")
    qval = benjamini_hochberg(pval)
    significant = (qval <= alpha_fdr)

    meta = dict(
        k=k, train_epochs=train_epochs, lr=lr,
        weight_align=weight_align, weight_margin=weight_margin, weight_lap=weight_lap,
        lam=lam, T=T, R_perm=R_perm, alpha_fdr=alpha_fdr, seed=seed,
        edge_feat_meta=ef_meta,
        device=str(dev), precision="fp32"
    )
    print("- Done!")
    return dict(
        K=K, rows=rows, cols=cols,
        scores=_to_numpy(scores),
        logit=logit, prob=prob, pval=pval, qval=qval, significant=significant,
        meta=meta
    )
