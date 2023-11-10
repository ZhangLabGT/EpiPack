import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
from typing import Literal
from tqdm import tqdm
from torch.autograd import Variable
from .net import EncoderAE, DecoderBinaryVAE
from .loss import bce_loss, mse_loss


class peak_dataset(Dataset):
    def __init__(self, counts_enhancer, batch=None):
        
        #check promoter/enhancer matrix
        assert not len(counts_enhancer) == 0, "Lack of the enhancer matrix"

        self.counts_enhancer = torch.FloatTensor(counts_enhancer)
        if batch:
            self.batch = torch.FloatTensor(batch)
        self.batch = None

    def __len__(self):
        return self.counts_enhancer.shape[0]

    def __getitem__(self, idx):
        sample = {"enhancer": self.counts_enhancer[idx,:]}
        return sample


class Peak_Model(nn.Module):
    def __init__(self, 
                 count_enhancer, 
                 layer_num: int = 2, 
                 batch_size: int = 512, 
                 hidden_dim: int = 256, 
                 dropout_rate: float = 0.1, 
                 z_dim: int = 50, 
                 device: Literal['auto','gpu','cpu'] = 'auto',
                 lib_size: bool = True,
                 region_factor: bool = False):

        super(Peak_Model, self).__init__()
 
        self.enhancer_count = count_enhancer
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.dropout_rate = dropout_rate
        self.z_dim = z_dim
        self.device = device
        self.lib_size = lib_size
        self.region_factor = region_factor

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

        ## promoter encoder
        self.Encoder = EncoderAE(
            n_input=self.enhancer_count.shape[1], 
            n_layers = self.layer_num, 
            n_output=self.z_dim, 
            n_hidden=self.hidden_dim, 
            dropout_rate=self.dropout_rate,
            use_layer_norm=True,
            use_batch_norm=False
        ).to(self.device_use)
        
        self.Decoder = DecoderBinaryVAE(
            n_input=self.z_dim, 
            n_layers = self.layer_num, 
            n_output=self.enhancer_count.shape[1], 
            n_hidden=self.hidden_dim, 
            dropout_rate=self.dropout_rate,
            use_layer_norm=True,
            use_batch_norm=False
        ).to(self.device_use)
        
        self.train_dataset = peak_dataset(counts_enhancer = count_enhancer)
        self.train_loader = DataLoader(dataset = self.train_dataset, batch_size = self.batch_size, shuffle = True)
        cutoff = 3000
        if len(self.train_dataset) > cutoff:
            sample_test = torch.randperm(n = len(self.train_dataset))[:cutoff]
            test_dataset = Subset(self.train_dataset, sample_test)
        self.test_loader = DataLoader(dataset = test_dataset, batch_size = len(test_dataset), shuffle = False)

        if self.region_factor:
            self.r_f = torch.nn.Parameter(torch.zeros(self.enhancer_count.shape[1])).to(self.device_use)

        print("- Model intialization completed.")

    
    def inference(self, x):

        z = self.Encoder(x)
        
        if self.lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
            
            return {'z':z, 'lib':library}
        
        return{'z':z, 'lib':None}
        

    
    def generative(self, z, lib = None):

        x_rate = self.Decoder(z = z, lib = lib)

        return {'x_rate': x_rate}


    def loss(self, count= None, rec= None):
            #change loss function
            #1. reconstruction loss
            f = torch.sigmoid(self.r_f) if self.region_factor else 1

            loss_bce = bce_loss(rec*f, (count>0).float()).sum()

            #loss_mse = mse_loss(rec*f, (count>0).float()).sum()
            return loss_bce
            
    
    def train_model(self, 
                    nepochs:int = 50, 
                    weight_decay:float = 5e-4, 
                    learning_rate:float = 1e-4):
        
        self.train()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        loop = tqdm(range(nepochs), total=nepochs, desc="Epochs")
        for epoch in loop:
            for id, x in enumerate(self.train_loader):
                
                dict_inf = self.inference(x['enhancer'].to(self.device_use))
                
                dict_gen = self.generative(z=dict_inf['z'].to(self.device_use), lib=dict_inf['lib'].to(self.device_use))

                loss = self.loss(count=x['enhancer'].to(self.device_use), rec=dict_gen['x_rate'].to(self.device_use))

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 10)
                self.optimizer.step()

                #if id == (interval_num-1):
                #    print("epoch:{}/{}, train_bce:{}".format(epoch, nepochs, loss))

            #test
            with torch.no_grad():
                for id, x in enumerate(self.test_loader):

                    dict_inf = self.inference(x['enhancer'].to(self.device_use))

                    dict_gen = self.generative(z=dict_inf['z'].to(self.device_use), lib=dict_inf['lib'].to(self.device_use))

                    loss_test = self.loss(count=x['enhancer'].to(self.device_use), rec=dict_gen['x_rate'].to(self.device_use))/self.enhancer_count.shape[0]

                    #print("epoch:{}/{}, test_bce:{}".format(epoch, nepochs, loss_test))

            loop.set_postfix(loss_val = loss_test.item())

        
    def get_z(self):
        self.eval()
        z_latent = self.Encoder(torch.FloatTensor(self.enhancer_count).to(self.device_use))

        return z_latent.cpu()
