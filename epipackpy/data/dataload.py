import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from os import PathLike

def read_10x_mtx(datapath: PathLike, multiomic: bool = False):
    '''
    read atac data from 10x .h5 file
    multiomic: if the dataset is multiomic data, use True.
    '''
    adata = sc.read_10x_mtx(datapath, gex_only=False)

    if multiomic:
        adata = adata[:, list(map(lambda x: x == "Peaks", adata.var["feature_types"]))]
    
    return adata


def read_10x_h5(datapath: PathLike, multiomic: bool = False):
    '''
    read atac data from 10x .h5 file
    multiomic: if the dataset is multiomic data, use True.
    '''
    adata = sc.read_10x_h5(datapath, gex_only=False)

    if multiomic:
        adata = adata[:, list(map(lambda x: x == "Peaks", adata.var["feature_types"]))]
    
    return adata
