import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mse_loss(decoder_out, x):
    '''
    decoder_out: Output of the decoder
    x: original matrix
    '''
    loss_rl = nn.MSELoss()
    result_mse = loss_rl(decoder_out, x)

    return result_mse

def bce_loss(decoder_out, x):
    loss_bce = nn.BCELoss(reduction='none')
    result_bce = loss_bce(decoder_out, x).sum(dim=-1)

    return result_bce

def KL_loss(p_pred,q_target):
    loss_kl = nn.KLDivLoss(log_target=True, reduction = 'batchmean')
    log_target = F.log_softmax(q_target, dim=1)
    result_kl = loss_kl(p_pred, q_target)

    return result_kl

def compute_pairwise_distances(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def _gaussian_kernel_matrix(x, y, device):
    sigmas = torch.tensor([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6], device = device)
    dist = compute_pairwise_distances(x, y)
    beta = 1. / (2. * sigmas[:,None])
    s = - beta.mm(dist.reshape((1, -1)) )
    result =  torch.sum(torch.exp(s), dim = 0)
    return result


def maximum_mean_discrepancy(xs, batch_ids, device, ref_batch = None): #Function to calculate MMD value
    # number of cells
    assert batch_ids.shape[0] == xs.shape[0]
    batches = torch.unique(batch_ids, sorted = True)
    nbatches = batches.shape[0]
    if ref_batch is None:
        # select the first batch, the batches are equal sizes
        ref_batch = batches[0]
    # assuming batch 0 is the reference batch
    cost = 0
    # within batch
    for batch in batches:
        xs_batch = xs[batch_ids == batch, :]
        if batch == ref_batch:
            cost += (nbatches - 1) * torch.mean(_gaussian_kernel_matrix(xs_batch, xs_batch, device))
        else:
            cost += torch.mean(_gaussian_kernel_matrix(xs_batch, xs_batch, device))
    
    # between batches
    xs_refbatch = xs[batch_ids == ref_batch]
    for batch in batches:
        if batch != ref_batch:
            xs_batch = xs[batch_ids == batch, :]
            cost -= 2.0 * torch.mean(_gaussian_kernel_matrix(xs_refbatch, xs_batch, device))
    
    cost = torch.sqrt(cost ** 2 + 1e-9)
    if cost.data.item()<0:
        cost = torch.tensor([0.0], device = device)

    return cost

def maximum_mean_discrepancy_transfer(xs, batch_ids, device, ref_batch = None): #Function to calculate MMD value
    # number of cells
    assert batch_ids.shape[0] == xs.shape[0]
    batches = torch.unique(batch_ids, sorted = True)
    nbatches = batches.shape[0]
    if ref_batch is None:
        # select the first batch, the batches are equal sizes
        ref_batch = batches[0]
    # assuming batch 0 is the reference batch
    cost = 0

    # within batch (without ref inner)
    for batch in batches:
        if batch != ref_batch:
            xs_batch = xs[batch_ids == batch, :]
            cost += torch.mean(_gaussian_kernel_matrix(xs_batch, xs_batch, device))

    # between batches
    xs_refbatch = xs[batch_ids == ref_batch]
    for batch in batches:
        if batch != ref_batch:
            xs_batch = xs[batch_ids == batch, :]
            cost -= 2.0 * torch.mean(_gaussian_kernel_matrix(xs_refbatch, xs_batch, device))

        
    cost = torch.sqrt(cost ** 2 + 1e-9)
    if cost.data.item()<0:
        cost = torch.tensor([0.0], device = device)

    return cost


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x)+np.inf, x)

def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def _nelem(x):
    nelem = torch.reduce_sum(torch.float(~torch.isnan(x), torch.float32))
    return torch.float(torch.where(torch.equal(nelem, 0.), 1., nelem), x.dtype)

def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return torch.divide(torch.reduce_sum(x), nelem)

class NB():
    """\
    Description:
    ------------
        The loss term of negative binomial
    Usage:
    ------------
        nb = NB(theta = theta, scale_factor = libsize, masking = False)
        nb_loss = nb.loss(y_true = mean_x, y_pred = x)        
    """
    def __init__(self, theta=None, masking=False, scope='nbinom_loss/',
                 scale_factor=1.0, debug=False, device = 'cpu'):
        """\
        Parameters:
        -----------
            theta: theta is the dispersion parameter of the negative binomial distribution. the output of the estimater
            scale_factor: scaling factor of y_pred (observed count), of the shape the same as the observed count. 
            scope: not sure
        """
        # for numerical stability, 1e-10 might not be enough, make it larger when the loss becomes nan
        self.eps = 1e-6
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope
        self.masking = masking
        self.theta = theta
        self.device = device

    def loss(self, y_true, y_pred, mean=True):
        """\
        Parameters:
        -----------
            y_true: the mean estimation. should be the output of the estimator
            y_pred: the observed counts.
            mean: calculate the mean of loss
        """
        scale_factor = self.scale_factor
        eps = self.eps
        
        y_true = y_true.type(torch.float32)
        y_pred = y_pred.type(torch.float32) * scale_factor

        if self.masking:
            nelem = _nelem(y_true)
            y_true = _nan2zero(y_true)

        # Clip theta
        # theta = torch.minimum(self.theta, torch.tensor(1e6).to(self.device))
        theta = self.theta.clamp(min = None, max = 1e6).to(self.device)

        t1 = torch.lgamma(theta+eps) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
        t2 = (theta+y_true) * torch.log(1.0 + (y_pred/(theta+eps))) + (y_true * (torch.log(theta+eps) - torch.log(y_pred+eps)))
        final = t1 + t2

        final = _nan2inf(final)

        if mean:
            if self.masking:
                final = torch.divide(torch.reduce_sum(final), nelem)
            else:
                final = torch.mean(final)

        return final  
    
class ZINB(NB):
    """\
    Description:
    ------------
        The loss term of zero inflated negative binomial (ZINB)
    Usage:
    ------------
        zinb = ZINB(pi = pi, theta = theta, scale_factor = libsize, ridge_lambda = 1e-5)
        zinb_loss = zinb.loss(y_true = mean_x, y_pred = x)
    """
    def __init__(self, pi, ridge_lambda=0.0, scope='zinb_loss/', **kwargs):
        """\
        Parameters:
        -----------
            pi: the zero-inflation parameter, the probability of a observed zero value not sampled from negative binomial distribution. Should be the output of the estimater
            ridge_lambda: ridge regularization for pi, not in the likelihood function, of the form ``ridge_lambda * ||pi||^2'', set to 0 if not needed.
            scope: not sure
            kwargs includes: 
                theta: theta is the dispersion parameter of the negative binomial distribution. the output of the estimater
                scale_factor: scaling factor of y_pred (observed count), of the shape the same as the observed count. 
                masking:
        """
        super().__init__(scope=scope, **kwargs)
        if not torch.is_tensor(pi):
            self.pi = torch.tensor(pi).to(self.device)
        else:
            self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        """\
        Parameters:
        -----------
            y_true: the mean estimation. should be the output of the estimator
            y_pred: the observed counts.
            mean: calculate the mean of loss
        """
        # set the scaling factor (libsize) for each observed (normalized) counts
        scale_factor = self.scale_factor
        # the margin for 0
        eps = self.eps

        # calculate the negative log-likelihood of nb distribution. reuse existing NB neg.log.lik.
        # mean is always False here, because everything is calculated
        # element-wise. we take the mean only in the end
        # -log((1-pi) * NB(x; mu, theta)) = - log(1 - pi) + nb.loss(x; mu, theta)
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log((torch.tensor(1.0+eps).to(self.device)-self.pi))
        y_true = y_true.type(torch.float32)
        
        # scale the observed (normalized) counts by the scaling factor
        y_pred = y_pred.type(torch.float32) * scale_factor
        # compute elementwise minimum between self.theta and 1e6, make sure all values are not inf
        # theta = torch.minimum(self.theta, torch.tensor(1e6).to(device))
        theta = self.theta.clamp(min = None, max = 1e6).to(self.device)

        # calculate the negative log-likelihood of the zero inflation part
        # first calculate zero_nb = (theta/(theta + x))^theta
        zero_nb = torch.pow(theta/(theta+y_pred+eps), theta)
        # then calculate the negative log-likelihood of the zero inflation part 
        # -log(pi + (1 - pi)*zero_nb)
        zero_case = -torch.log(self.pi + ((1.0-self.pi)*zero_nb)+eps)
        # when observation is 0, negative log likelihood equals to zero_case, or equals to nb_case
        # result = torch.where(torch.less(y_true, 1e-8), zero_case, nb_case)
        result = torch.where(y_true < 1e-8, zero_case, nb_case)

        # regularization term
        ridge = self.ridge_lambda*torch.square(self.pi)
        result += ridge

        # calculate the mean of all likelihood over genes and cells
        if mean:
            if self.masking:
                result = _reduce_mean(result) 
            else:
                result = torch.mean(result)

        result = _nan2inf(result)
        
        return result


### supervised contrastive loss
def contrastive_loss(representations, label, T=0.5, device = None):

    n = label.shape[0]

    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))

    mask_no_sim = torch.ones_like(mask) - mask

    mask_dui_jiao_0 = torch.ones(n ,n) - torch.eye(n, n)

    similarity_matrix = torch.exp(similarity_matrix/T)

    similarity_matrix = similarity_matrix*mask_dui_jiao_0.to(device)

    sim = mask*similarity_matrix

    no_sim = similarity_matrix - sim

    no_sim_sum = torch.sum(no_sim , dim=1)

    no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
    sim_sum  = sim + no_sim_sum_expend
    loss = torch.div(sim , sim_sum)

    loss = mask_no_sim + loss + torch.eye(n, n )


    loss = -torch.log(loss) 
    loss = torch.sum(torch.sum(loss, dim=1) )/(2*n)

    return loss