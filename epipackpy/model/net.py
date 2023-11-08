from typing import Iterable
import torch
from torch import nn as nn
import collections
import torch.nn.functional as F
from .utils import cosine_dist

def identity(x):
    return x

def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)

class FCLayers(nn.Module):
    """
    A helper class to build fully-connected layers for a neural network.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        use_activation: bool = False,
        bias: bool = True,
        inject_covariates: bool = False,
        activation_fn: nn.Module = nn.LeakyReLU,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []
        
        # self.n_cat_list = [n_batches]
        # cat_dim = n_batches
        cat_dim = sum(self.n_cat_list)
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(
                                # inject cat_dim into the first layer if deeply_inject_covariates is False
                                # to all dimensions if the deeply_inject_covariates is True
                                n_in + cat_dim * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def set_online_update_hooks(self, hook_first_layer=True):
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def forward(self, x: torch.Tensor, *cat_list: int):
        """
        Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        x: torch.Tensor

        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``

        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError(
                "nb. categorical args provided doesn't match init. params."
            )
        # self.n_cat_list = [n_batches], cat is a int indicating the current batch
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x


######################################
#         Encoder materials          #
###################################### 

class Encoder(nn.Module):
    """
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        When `None`, defaults to `torch.exp`.
    return_dist
        If `True`, returns directly the distribution of z instead of its parameters.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        inject_covariates: bool = False,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.fc = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list, # [n_batches]
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_activation=True,
            bias=True,
            activation_fn=nn.LeakyReLU,
        )

        self.mean_layer = nn.Linear(n_hidden, n_output)
        self.var_layer = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""
        The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for latent distribution
        q = self.fc(x, *cat_list)
        mu = self.mean_layer(q)
        var = self.var_layer(q)
        return mu, var
    

class EncoderAE(nn.Module):
    """
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        When `None`, defaults to `torch.exp`.
    return_dist
        If `True`, returns directly the distribution of z instead of its parameters.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        inject_covariates: bool = False,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.fc = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list, # [n_batches]
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_activation=True,
            bias=True,
            activation_fn=nn.LeakyReLU,
        )

        self.mean_layer = nn.Linear(n_hidden, n_output)
        #self.var_layer = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""
        The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for latent distribution
        q = self.fc(x, *cat_list)
        mu = self.mean_layer(q)
        #var = self.var_layer(q)
        return mu#, var



#############################
#       classifier          #
#############################


# classifier projection function
class classifier_layer(nn.Module):
    """
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 50,
        dropout_rate: float = 0.1,
        inject_covariates: bool = False,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False
    ):
        super().__init__()

        self.fc = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list, # [n_batches]
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_activation=True,
            bias=True,
            activation_fn=nn.LeakyReLU,
        )

        self.mean_layer = nn.Linear(n_hidden, n_output)


    def forward(self, x: torch.Tensor, *cat_list: int):
        """
        The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for latent distribution
        q = self.fc(x, *cat_list)
        mu = self.mean_layer(q)
        return mu#, var
    

class CosCell(nn.Module):
    """
    
    Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin

    """

    def __init__(self, in_features, out_features, s=30.0):
        super(CosCell, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)


    def forward(self, input):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        output = self.s * cosine

        return output
    
class InnerCos(nn.Module):

    def __init__(self, in_feature, n_class):
        super(InnerCos, self).__init__()
        self.in_feature = in_feature
        self.n_class = n_class
        self.centers = nn.Parameter(torch.randn(self.n_class, self.in_feature))

    def get_centers(self):
        """Returns estimated centers"""
        return self.centers

    def forward(self, ref_emb, ref_label):
        center = self.centers[ref_label]
        cos_mat = cosine_dist(ref_emb, center)
        cos_mat_diag = torch.diag(cos_mat)
        cos_dist = torch.clamp(cos_mat_diag, min=1e-12, max=1e+12).mean(dim=-1)

        return cos_dist

#### re-implement
class InnerCosLoss(nn.Module):
    """Implements the Center loss from https://ydwen.github.io/papers/WenECCV16.pdf"""
    def __init__(self, in_feature, n_class):
        super(InnerCosLoss, self).__init__()
        self.in_feature = in_feature
        self.n_class = n_class
        self.centers = nn.Parameter(torch.randn(self.n_class, self.in_feature))

    def get_centers(self):
        """Returns estimated centers"""
        return self.centers

    def forward(self, ref_emb, ref_label):
        features = F.normalize(ref_emb)
        batch_size = len(ref_label)
        
        self.centers.data = F.normalize(self.centers.data, p=2, dim=1)

        centers_batch = self.centers[ref_label]

        cos_sim = nn.CosineSimilarity()
        cos_diff = 1. - cos_sim(features, centers_batch)
        center_loss = torch.sum(cos_diff) / batch_size
                
        return center_loss
    
## center loss
class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, center_init=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.centers = nn.Parameter(torch.zeros(self.num_classes, self.feat_dim), requires_grad=True)
        self.centers.data = torch.from_numpy(center_init).float()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)

        ##inner
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(self.centers, self.centers.t(), beta=1, alpha=-2)
        
        ##center
        distmat = torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(self.centers, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask_in = labels.eq(classes.expand(batch_size, self.num_classes))
        mask_out = labels.ne(classes.expand(batch_size, self.num_classes))

        dist_in = distmat * mask_in.float()
        loss_inner = dist_in.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        dist_out = distmat * mask_out.float()
        loss_inter = dist_out.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss_inner, loss_inter
    
######################################
#         Decoder materials          #
###################################### 

# Decoder poisson test
class DecoderPoissonVAE(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        inject_covariates: bool = False,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False
    ):
        super().__init__()
        self.fc = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_activation=True,
            bias=True,
            activation_fn=nn.ReLU,
        )

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output, bias=True), 
            nn.Softmax(dim=-1)
        )

    def forward(self, z: torch.Tensor, lib_size_factor: torch.Tensor, *cat_list: int):
        # The decoder returns values for the parameters of the ZINB distribution
        x = self.fc(z, *cat_list)
        px_scale = self.px_scale_decoder(x)

        px_rate = torch.exp(lib_size_factor) * px_scale
        
        return px_rate

# Decoder peak model
class DecoderBinaryVAE(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        inject_covariates: bool = False,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False
    ):
        super().__init__()
        self.fc = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_activation=True,
            bias=True,
            activation_fn=nn.LeakyReLU,
        )

        self.n_in = n_input
        self.n_out = n_output
        self.output = nn.Sequential(
            nn.Linear(n_hidden, n_output, bias=True)
            )
        
        self.ber_activation = nn.Sigmoid()

    def forward(self, z: torch.Tensor, lib: torch.Tensor, *cat_list: int):
        # The decoder returns values for the parameters of the ZINB distribution
        x = self.fc(z, *cat_list)
        x_ = self.output(x)
        if lib is not None:
            x_rate = self.ber_activation(x_ + torch.logit(torch.exp(lib)/self.n_out,eps = 1e-7))
        else:
            x_rate = self.ber_activation(x_)

        return x_rate #, pi, theta
    
# Decoder bridge VAE model
class DecoderVAE(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        inject_covariates: bool = False,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False
    ):
        super().__init__()
        self.fc = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_activation=True,
            bias=True,
            activation_fn=nn.LeakyReLU,
        )

        self.output = OutputLayer(features = [n_hidden, n_output])

    def forward(self, z: torch.Tensor, *cat_list: int):
        # The decoder returns values for the parameters of the ZINB distribution
        x = self.fc(z, *cat_list)
        mu, pi, theta = self.output(x)
        return mu, pi, theta
    
class OutputLayer(nn.Module):
    def __init__(self, features, zero_inflation = True) -> None:
        super().__init__()
        self.output_size = features[1]
        self.last_hidden = features[0]
        self.zero_inflation = zero_inflation
        self.mean_layer = nn.Sequential(nn.Linear(self.last_hidden, self.output_size), nn.ReLU())
        # ! Parameter Pi needs Sigmoid as activation func
        if self.zero_inflation: 
            self.pi_layer = nn.Sequential(nn.Linear(self.last_hidden, self.output_size), nn.Sigmoid())
        self.theta_layer = nn.Sequential(nn.Linear(self.last_hidden, self.output_size),nn.Softplus())

    def forward(self, decodedData):
        mu = self.mean_layer(decodedData)
        theta = self.theta_layer(decodedData)
        if self.zero_inflation:
            pi = self.pi_layer(decodedData)
            return mu, pi, theta
        else:
            return mu, theta

class Pseudopoint_Layer(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(n_input, n_output, bias=False),
            nn.Hardtanh(0.0, 1.0)
        )

    def forward(self, x):

        output = self.layer(x)

        return output

#class EarlyStopping:
#    def __init__(self, patience=10, verbose=False):
#        """
#        Args:
#            patience (int): How long to wait after last time loss improved.
#                            Default: 10
#            verbose (bool): If True, prints a message for each loss improvement. 
#                            Default: False
#        """
#        self.patience = patience
#        self.verbose = verbose
#        self.best_score = None
#        self.counter = 0
#        self.loss_min = np.Inf
#        self.model_para = None
#
#    def __call__(self, loss, model):
#
#        if self.best_score = None:
#            self.best_score = loss
#            self.model_para = model
#
#       elif loss >= self.best_score:
#            self.counter += 1
#            if self.verbose:
#                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#            if self.counter >= self.patience:
#                self.early_stop = True
#        else:
#            self.best_score = loss
#            self.counter = 0
#            self.model_para = model.state_dict()