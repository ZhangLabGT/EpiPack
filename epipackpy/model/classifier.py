import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import Literal
from tqdm import tqdm
from .net import classifier_layer, CosCell, InnerCos, InnerCosLoss, CenterLoss
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from .loss import contrastive_loss
from .utils import cosine_dist


class classifier_dataset(Dataset):
    def __init__(self, ref_emb, ref_label = None, train=None):
        
        #check promoter/enhancer matrix
        assert not len(ref_emb) == 0, "Lack of the enhancer matrix"

        self.ref_emb = torch.FloatTensor(ref_emb)
        self.train = train
        if self.train:
            self.ref_label = torch.IntTensor(ref_label)

    def __len__(self):
        return self.ref_emb.shape[0]

    def __getitem__(self, idx):
        if self.train:
            sample = {"ref_emb": self.ref_emb[idx,:], "ref_label":self.ref_label[idx]}
        else:
            sample = {"ref_emb": self.ref_emb[idx,:]}
        return sample



class Classifier(nn.Module):
    def __init__(self, 
                 ref_latent_emb,  
                 ref_label,
                 class_num,
                 hidden_num,
                 z_dim: int = 30,  
                 dropout_rate: float = 0.1,
                 layer_num: int = 1, 
                 batch_size: int = 128, 
                 batchnorm: bool = True,
                 device: Literal['auto','gpu','cpu'] = 'auto'):

        super(Classifier, self).__init__()
 
        self.ref_latent_emb = ref_latent_emb
        self.ref_label = ref_label
        self.class_num = class_num
        self.hidden_num = hidden_num
        self.batch_size = batch_size
        self.layer_num = layer_num
        self.dropout_rate = dropout_rate
        self.z_dim = z_dim
        self.device = device
        self.batchnorm = batchnorm

        print("- Classifier initializing...")

        assert self.ref_latent_emb.shape[0] == len(self.ref_label), "Reference label set has different cell numbers with reference embedding."

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
        self.Encoder = classifier_layer(
            n_input=self.ref_latent_emb.shape[1], 
            n_layers = self.layer_num, 
            n_output = self.z_dim, 
            n_hidden = self.hidden_num,
            dropout_rate= self.dropout_rate,
            use_layer_norm = False,
            use_batch_norm = self.batchnorm
            ).to(self.device_use)

        #self.Coscell = CosCell(self.z_dim, self.class_num).to(self.device_use)
        #self.Coscell = nn.Linear(self.z_dim, self.class_num).to(self.device_use)

        self.class_mu = []
        for i in np.unique(self.ref_label):
            mu_class = self.ref_latent_emb[np.where(self.ref_label==i)]
            self.class_mu.append(np.mean(mu_class, axis=0).tolist())

        self.class_mu = np.array(self.class_mu)
    
        ## split train test
        #X_train, X_test, y_train, y_test = train_test_split(self.ref_latent_emb, self.ref_label, test_size = 0.2, stratify=self.ref_label)
        self.train_dataset = classifier_dataset(self.ref_latent_emb, self.ref_label, train=True)
        #self.test_dataset = classifier_dataset(X_test, y_test, train=False)
        self.train_loader = DataLoader(dataset = self.train_dataset, batch_size = self.batch_size, shuffle = True)
        #self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)

        print("- Classifier intialization completed.")
    
    def inference(self, x):

        out = self.Encoder(x)
        #z = self.Coscell(out)
        
        return{"cls_latent": out}
            
    
    def train_model(self, 
                    nepochs:int = 50, 
                    weight_decay:float = 5e-4, 
                    learning_rate:float = 1e-4,
                    #inner_reg:float = 1,
                    optim: Literal = ['Adam', 'SGD']):
        
        self.train()
        
        #criterion = torch.nn.CrossEntropyLoss().to(self.device_use)
        #self.criterion_inner = CenterLoss(self.class_num, self.z_dim).to(self.device_use)

        #optimizer_main = torch.optim.Adam(self.Encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #optimizer_inter = torch.optim.Adam(self.Coscell.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.Encoder.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay)
            #self.optimizer_inner  = torch.optim.Adam(self.criterion_inner.parameters(),
            #                    lr=learning_rate,
            #                    weight_decay=weight_decay)
        elif optim == 'SGD':
            self.optimizer = torch.optim.SGD([{'params': self.Encoder.parameters()}, {'params': self.Coscell.parameters()}],
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=weight_decay)
            #self.optimizer_inner  = torch.optim.SGD(self.criterion_inner.parameters(),
            #                    lr=learning_rate,
            #                    momentum=0.9,
            #                    weight_decay=weight_decay)

        loop = tqdm(range(nepochs), total=nepochs, desc="Epochs")
        for epoch in loop:
            for id, x in enumerate(self.train_loader):
                
                dict_inf = self.inference(x['ref_emb'].to(self.device_use))

                loss = contrastive_loss(dict_inf['cls_latent'], x["ref_label"].long().to(self.device_use))
                #loss_inner = self.criterion_inner(dict_inf['cls_latent'], x["ref_label"].long().to(self.device_use))
                #loss_inner = inner_reg

                #loss = loss_inter + inner_reg*loss_inner

                # Backward and optimize
                self.optimizer.zero_grad()
                #self.optimizer_inner.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 10)
                self.optimizer.step()

                #for param in self.criterion_inner.parameters():
                #    param.grad.data *= (1. / inner_reg)

                #self.optimizer_inner.step()

                #if id == (interval_num-1):
                #    print("epoch:{}/{}, train_bce:{}".format(epoch, nepochs, loss))

            #test
            #with torch.no_grad():
            #    for id, x in enumerate(self.test_loader):

            #        dict_inf = self.inference(x['ref_emb'].to(self.device_use))
            #        loss_test= criterion(dict_inf['logit'], x["ref_label"].long().to(self.device_use))

            #loop.set_postfix(loss_inter = loss_inter.item(), loss_inner = loss_inner.item())
            loop.set_postfix(loss_inter = loss.item())

        
    def get_z(self, input):
        self.eval()
        z_latent = self.inference(torch.FloatTensor(input).to(self.device_use))

        return z_latent
    
    #def get_dis(self, input):
    #    query_emb = torch.FloatTensor(input).to(self.device_use)
    #    ctrs = self.criterion_inner.get_centers()
    #
    #    cos_diff = cosine_dist(query_emb, ctrs)
    #
    #    return cos_diff
        
