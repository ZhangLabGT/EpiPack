import pandas as pd
import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
from torch.autograd import Variable
from typing import Literal
from sklearn.mixture import GaussianMixture
from .net import Encoder, EncoderAE, DecoderVAE, DecoderBinaryVAE, Pseudopoint_Layer, DecoderPoissonVAE
from .loss import mse_loss, maximum_mean_discrepancy_transfer,maximum_mean_discrepancy, NB, ZINB
from tqdm import tqdm


class model_dataset2(Dataset):
    def __init__(self, counts_promoter, z_enhancer, batch_id):
        
        #check promoter/enhancer matrix
        assert not len(z_enhancer) == 0, "Lack of the peak embedding matrix"
        assert not len(counts_promoter) == 0, "Lack of the genescore matrix"
        assert not len(batch_id) == 0, "Lack of the batch id set"

        self.counts_promoter = torch.FloatTensor(counts_promoter)
        self.z_enhancer = torch.FloatTensor(z_enhancer)
        self.batch_id = torch.IntTensor(batch_id)
    
    def __len__(self):
        return self.counts_promoter.shape[0]

    def __getitem__(self, idx):
        sample = {'promoter': self.counts_promoter[idx,:], 'enhancer': self.z_enhancer[idx,:], 'batch_id': self.batch_id[idx]}
        return sample


class VAE_MAPPING(nn.Module):
    '''
    This test model is for VAE GENE SCORE REGULARIZED BY PEAK EMBEDDING
    '''
    def __init__(self, promoter_dt, 
                 enhancer_z, 
                 batch_id, 
                 ref_embedding,
                 layer_num = 2, 
                 batch_size=256, 
                 hidden_dim=256, 
                 dropout_rate = 0, 
                 z_dim=50, 
                 reg_mmd=1,
                 reg_mmd_inter = 0.1, 
                 reg_kl=1e-10,
                 reg_rec=1,
                 reg_z_l2=0.01,
                 prior = "standard", #['standard', 'GMM', 'VAMP']
                 n_center = 50,
                 n_pseudopoint = 300,
                 device: Literal['auto','gpu','cpu'] = 'auto',
                 use_layer_norm: bool = False,
                 use_batch_norm: bool = True):

        super().__init__()

        self.promoter_dt = promoter_dt
        self.enhancer_z = enhancer_z
        self.ref_embedding = ref_embedding
        self.batch_id = batch_id
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.dropout_rate = dropout_rate
        self.z_dim = z_dim
        self.prior = prior
        self.n_c = n_center
        self.n_pesudopoint = n_pseudopoint
        self.device = device
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm

        # regularization parameters
        self.reg = {"mmd": reg_mmd, "kl": reg_kl, "rec": reg_rec, "z_l2": reg_z_l2, "mmd_inter": reg_mmd_inter} #epoch zone 1
                    #"mmd_2": reg_mmd_2, "rec_2": reg_rec_2, "z_l2_2": reg_z_l2_2} #epoch zone 2

        # Batch number: batch id should be a numpy list
        self.uniq_batch_id = [i for i in np.unique(batch_id)]

        print("- Model initializing...")

        #model initialization
        if self.device == 'auto':
            self.device_use = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                print("- [Auto-detection]: GPU detetced. Model will be initialized in GPU mode.")
            else:
                print("- [Auto-detection]: GPU not detected. Model will be initialized in CPU mode.")
        elif self.device == 'gpu':
            self.device_use = torch.device("cuda:0")
            print("- [Manually setting]: Model will be initialized in GPU mode.")
        elif self.device == 'cpu':
            self.device_use = torch.device("cpu")
            print("- [Manually setting]: Model will be initialized in CPU mode.")

        #model initialization
        ## promoter encoder
        self.Encoder = Encoder(n_input=self.promoter_dt.shape[1], 
                                 n_layers = self.layer_num, 
                                 n_output=self.z_dim, 
                                 n_hidden=self.hidden_dim, 
                                 dropout_rate=self.dropout_rate,
                                 n_cat_list=[len(self.uniq_batch_id)],
                                 use_layer_norm=self.use_layer_norm,
                                 use_batch_norm=self.use_batch_norm
                                 ).to(self.device_use)
         
        self.Decoder = DecoderVAE(n_input=self.z_dim, 
                                 n_layers = self.layer_num, 
                                 n_output=self.promoter_dt.shape[1], 
                                 n_hidden=self.hidden_dim, 
                                 dropout_rate=self.dropout_rate,
                                 n_cat_list=[len(self.uniq_batch_id)],
                                 use_layer_norm=self.use_layer_norm,
                                 use_batch_norm=self.use_batch_norm
                                 ).to(self.device_use)       

        
        self.train_dataset = model_dataset2(counts_promoter = promoter_dt, z_enhancer = enhancer_z, batch_id = batch_id)
        self.train_loader = DataLoader(dataset = self.train_dataset, batch_size = self.batch_size, shuffle = True)

        ## cut off test dataset scale
        cutoff = 1000
        if len(self.train_dataset) > cutoff:
            sample_test = torch.randperm(n = len(self.train_dataset))[:cutoff]
            test_dataset = Subset(self.train_dataset, sample_test)
        self.test_loader = DataLoader(dataset = test_dataset, batch_size = len(test_dataset), shuffle = False)

        
        # model initialization
        if self.prior == 'GMM':
            print("- Initializing Gaussian Mixture parameters and network parameters...")

            pi_dict = []
            mu_dict = []
            var_dict = []

            #init gmm para
            gmm = GaussianMixture(n_components=self.n_c, covariance_type='diag')
            gmm.fit(enhancer_z)

            #for i in  

            self.pi = nn.Parameter(torch.ones(self.n_c)/self.n_c, requires_grad=True)
            self.mu = nn.Parameter(torch.zeros(self.z_dim, self.n_c), requires_grad=True)
            self.var = nn.Parameter(torch.ones(self.z_dim, self.n_c), requires_grad=True)
        
            self.pi.data = torch.from_numpy(gmm.weights_.T).float()
            self.mu.data = torch.from_numpy(gmm.means_.T).float()
            self.var.data = torch.log(torch.from_numpy(gmm.covariances_.T).float())

            print("- Initialization complete...")
            
        elif self.prior == 'standard':

            print("- Initialization complete...")

    
    def reparametrize(self, mu, logvar, clamp = 0):
        # exp(0.5*log_var) = exp(log(\sqrt{var})) = \sqrt{var}
        std = logvar.mul(0.5).exp_()
        if clamp > 0:
            # prevent the shrinkage of variance
            std = torch.clamp(std, min = clamp)

        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).to(self.device_use)
        return eps.mul(std).add_(mu)

    def inference(self, m_promoter, batch_id = None, clamp_promoter = 0.0, eval=False, print_stat = False):

        #check batch id size
        assert m_promoter.shape[0] == batch_id.shape[0]

        library = torch.log(m_promoter.sum(1)).unsqueeze(1)
        #library = torch.ones(_library.shape)
        x_ = torch.log(1 + m_promoter)

        mu_p, logvar_p = self.Encoder(x_, batch_id)


        if not eval:
            z_p = self.reparametrize(mu = mu_p, logvar = logvar_p, clamp=clamp_promoter)

            if print_stat:
                print("mean z_p: {:.5f}".format(torch.mean(mu_p).item()))
                print("mean var z_p: {:.5f}".format(torch.mean(logvar_p.mul(0.5).exp_()).item()))
                
            return {"mu_p":mu_p, "logvar_p": logvar_p, "z_p": z_p, "lib_size": library}

        return {"mu_p":mu_p, "logvar_p": logvar_p, "lib_size": library}
    
    def generative(self, z_p, batch_id):

        #decoder promoter
        mu, pi, theta = self.Decoder(z_p, batch_id)

        return {'rec_promoter':mu,'pi':pi, 'theta':theta}

    def get_gamma(self, z):
        """
        Inference c from z

        gamma is q(c|x)
        q(c|x) = p(c|z) = p(c)p(c|z)/p(z)
        """

        N = z.size(0)
        z = z.unsqueeze(2).expand(z.size(0), z.size(1), self.n_c)
        pi = self.pi.repeat(N, 1) # NxK
        #pi = torch.clamp(self.pi.repeat(N,1), 1e-10, 1) # NxK
        mu = self.mu.repeat(N,1,1) # NxDxK
        var = self.var.repeat(N,1,1) # NxDxK

        # p(c,z) = p(c)*p(z|c) as p_c_z
        p_c_z = torch.exp(torch.log(pi) - torch.sum(0.5*torch.log(2*math.pi*torch.exp(var)) + (z-mu).pow(2)/(2*torch.exp(var)), dim=1)) + 1e-10
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma
    
    
    def loss(self, dict_inf = None, count= None, rec= None, ref_z = None, batch_id = None, count_eh = None, lib_size = 1, rec_type = 'MSE'):
        '''
        Loss #1 + #2 = ELBO
        Loss #3 for regularization of the latent space z between "gene score" and "peak"
        Loss #4 MMD loss
        '''
        #change loss function
        # 1.kl divergence
        if self.prior == 'standard':

            kl_div = torch.sum(dict_inf["mu_p"].pow(2).add_(dict_inf["logvar_p"].exp()).mul_(-1).add_(1).add_(dict_inf["logvar_p"])).mul_(-0.5) 

        elif self.prior == 'GMM':

            #print('GMM loss in use')
            
            pi = self.pi
            logvar_c = self.var #+ 1e-10
            mu_c = self.mu

            mu_expand = dict_inf["mu_p"].unsqueeze(2).expand(dict_inf["mu_p"].size(0), dict_inf["mu_p"].size(1), self.n_c)
            logvar_expand = dict_inf["logvar_p"].unsqueeze(2).expand(dict_inf["logvar_p"].size(0), dict_inf["logvar_p"].size(1), self.n_c)
            #gamma
            gamma = self.get_gamma(dict_inf["z_p"])

            # log p(z|c)
            logpzc = -0.5*torch.sum(gamma*torch.sum(math.log(2*math.pi) + 
                                                    logvar_c + 
                                                    torch.exp(logvar_expand - logvar_c) + 
                                                    (mu_expand-mu_c).pow(2)/torch.exp(logvar_c), dim=1), dim=1)
    
            # log p(c)
            logpc = torch.sum(gamma*torch.log(pi), 1)

            # log q(z|x) or q entropy    
            qentropy = -0.5*torch.sum(1+dict_inf["logvar_p"]+math.log(2*math.pi), 1)

            # log q(c|x)
            logqcx = torch.sum(gamma*torch.log(gamma), 1)

            kl = -logpzc - logpc +qentropy + logqcx
            kl_div = torch.sum(kl)/self.batch_size

        #2. promoter rec loss
        if rec_type == 'MSE':
            rec_rate = rec['rec_promoter']*lib_size
            loss_rec = mse_loss(rec_rate, count.to(self.device_use))
        elif rec_type == 'NB':
            rec_rate = rec['rec_promoter']*lib_size
            loss_rec = NB(theta=rec['theta'], device=self.device_use).loss(y_true=count.to(self.device_use), y_pred=rec_rate)
        elif rec_type == 'ZINB':
            lamb_pi = 1e-5
            rec_rate = rec['rec_promoter']*lib_size
            loss_rec = ZINB(pi = rec["pi"], theta = rec["theta"], ridge_lambda = lamb_pi).loss(y_true=count.to(self.device_use), y_pred=rec_rate)
        else:
            raise ValueError("recon_loss can only be 'ZINB', 'NB', and 'MSE'")

        #3. promoter z L2 loss
        loss_z_l2 = mse_loss(dict_inf['z_p'], count_eh)

        #4. mmd loss
        #batch_ids = torch.tensor([1]*dict_inf['z_p'].shape[0] + [2]*dict_inf['z_p'].shape[0]).to(self.device_use)
        #max_batch_id = max(self.uniq_batch_id)
        #ref_id = torch.tensor([max_batch_id+1]*self.batch_size).to(self.device_use)
        #batch_ids = torch.concat((batch_id, ref_id))

        #random sample
        #xs = torch.cat((dict_inf['z_p'], ref_z))
        #ref = max_batch_id+1

        loss_mmd = maximum_mean_discrepancy(dict_inf['z_p'], batch_ids=batch_id, device=self.device_use)

        #max_batch_id = max(self.uniq_batch_id)

        #5. mmd inter
        ref_id = torch.tensor([0]*self.batch_size).to(self.device_use)
        query_id = torch.tensor([1]*len(dict_inf['z_p'])).to(self.device_use)
        batch_ids = torch.concat((ref_id, query_id))
        xs = torch.cat((dict_inf['z_p'], ref_z))

        loss_mmd_inter = maximum_mean_discrepancy(xs, batch_ids=batch_ids, device=self.device_use)

        return loss_z_l2, loss_rec, kl_div, loss_mmd, loss_mmd_inter

        
    def train_model(self, 
                    nepochs = 50, 
                    clamp = 0.0, 
                    weight_decay:float = 5e-4, 
                    learning_rate:float = 1e-4,
                    rec_loss="MSE"):

        self.train()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate, weight_decay=weight_decay)
        
        loop = tqdm(range(nepochs), total=nepochs, desc="Epochs")
        for epoch in loop:
            for id, x in enumerate(self.train_loader):

                #pass encoder
                dict_inf = self.inference(m_promoter = x['promoter'].to(self.device_use), 
                                          batch_id=x['batch_id'][:,None].to(self.device_use))
                
                #reconstruct gene score
                dict_gen = self.generative(z_p = dict_inf['z_p'].to(self.device_use), 
                                           batch_id=x['batch_id'][:,None].to(self.device_use))

                #reference sampling
                sample_index = np.random.choice(range(len(self.ref_embedding)), self.batch_size) 
                ref_z = torch.FloatTensor(self.ref_embedding[sample_index])

                #loss
                loss_z_l2, loss_promoter, loss_kl, loss_mmd, loss_mmd_inter = self.loss(rec = dict_gen, 
                                                                        dict_inf = dict_inf,
                                                                        ref_z=ref_z.to(self.device_use),
                                                                        count = x['promoter'].to(self.device_use), 
                                                                        batch_id=x['batch_id'].to(self.device_use),
                                                                        count_eh = x['enhancer'].to(self.device_use), 
                                                                        lib_size=dict_inf['lib_size'].to(self.device_use),
                                                                        rec_type=rec_loss)

                loss = self.reg['rec']*loss_promoter + self.reg['z_l2']*loss_z_l2 + self.reg['kl']*loss_kl + self.reg['mmd']*loss_mmd +self.reg['mmd_inter']*loss_mmd_inter
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
                self.optimizer.step()

                #else:
                #    loss = self.reg['rec_2']*loss_promoter + self.reg['z_l2_2']*loss_z_l2 + self.reg['kl']*loss_kl + self.reg['mmd_2']*loss_mmd
                #    self.optimizer.zero_grad()
                #    loss.backward()
                #    self.optimizer.step()
                    

                #if id == (interval_num-1):
                #    print("epoch:{}/{}, train_l2_z:{}, train_loss_rec:{}, train_kl_loss:{}, kl_reg:{}, train_mmd_loss:{}, mmd_reg:{}".format(epoch+1, nepochs, loss_z_l2, loss_promoter, loss_kl, self.reg['kl'], loss_mmd, self.reg['mmd']))
            with torch.no_grad():
                for id, x in enumerate(self.test_loader):

                    #pass encoder
                    dict_inf = self.inference(m_promoter = x['promoter'].to(self.device_use), 
                                            batch_id=x['batch_id'][:,None].to(self.device_use))
                
                    #reconstruct gene score
                    dict_gen = self.generative(z_p = dict_inf['z_p'].to(self.device_use), 
                                            batch_id=x['batch_id'][:,None].to(self.device_use))

                    
                    sample_index = np.random.choice(range(len(self.ref_embedding)), self.batch_size)
                    ref_z = torch.FloatTensor(self.ref_embedding[sample_index])
                    #loss
                    loss_z_l2, loss_promoter, loss_kl, loss_mmd, loss_mmd_inter = self.loss(rec = dict_gen, 
                                                                        dict_inf = dict_inf, 
                                                                        ref_z=ref_z.to(self.device_use),
                                                                        count = x['promoter'].to(self.device_use), 
                                                                        batch_id=x['batch_id'].to(self.device_use),
                                                                        count_eh = x['enhancer'].to(self.device_use), 
                                                                        lib_size=dict_inf['lib_size'].to(self.device_use),
                                                                        rec_type=rec_loss)
                        
                    #loss_test = self.reg['rec']*loss_promoter + self.reg['z_l2']*loss_z_l2 + self.reg['kl']*loss_kl + self.reg['mmd']*loss_mmd

                        #else:
                        #    loss_test = self.reg['rec_2']*loss_promoter + self.reg['z_l2_2']*loss_z_l2 + self.reg['kl']*loss_kl + self.reg['mmd_2']*loss_mmd
                        #
                        #    print("epoch:{}/{}, test_loss:{:.4f}".format(epoch, nepochs, loss_test.item()))
                        #    info = [
                        #        'rec loss: {:.4f}, MSE reg: {}'.format(loss_promoter.item(), self.reg['rec_2']),
                        #        'L2 z loss: {:.4f}, L2 reg: {}'.format(loss_z_l2.item(), self.reg['z_l2_2']),
                        #        'MMD loss: {:.4f}, MMD reg: {}'.format(loss_mmd.item(), self.reg['mmd_2']),
                        #        'KL loss: {:.4f}, KL reg: {}'.format(loss_kl.item(), self.reg['kl']),
                        #    ]
                    
                    loop.set_postfix(val_rec_loss = loss_promoter.item(), 
                                     val_kl_loss = loss_kl.item(), 
                                     val_loss_z_dist = loss_z_l2.item(), 
                                     val_mmd = loss_mmd.item(),
                                     val_mmd_inter = loss_mmd_inter.item())


    def get_latent(self):

        self.eval()

        dict_inf = self.inference(m_promoter = torch.FloatTensor(self.promoter_dt).to(self.device_use), 
                                  batch_id=torch.FloatTensor(self.batch_id)[:,None].to(self.device_use))

        return dict_inf['z_p'].cpu()
