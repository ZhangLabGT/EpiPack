import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import sklearn.preprocessing as sp

import polars as pl
import gzip
import pybiomart as pbm
from os import PathLike
from typing import Literal

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

def make_binary(adata: AnnData):
    '''
    convert raw peak-by-cell matrix into binary matrix
    raw data is saved at Anndata.layers['raw']
    '''
    adata.layers['raw'] = adata.X
    adata.X = sp.binarize(adata.X)

    return adata

def add_fragment_file(adata, 
                      frag_path: PathLike, 
                      genome:Literal ['Hg38','Hg19','mm39', 'mm10']):
    '''
    add fragment file to the peak anndata
    adata: peak anndata
    frag_path: fragment file path
    genome: add genome version
    '''
    
    if genome is None:
        raise ValueError(
            f'Genome version is not defined.'
        )

    print('- Creating fragment dataframe...')

    open_fn = gzip.open if frag_path.endswith(".gz") else open

    skip_rows = 0
    nbr_columns = 0
    with open_fn(frag_path,"rt") as frag_tsv:
        for line in frag_tsv:
            # Remove newlines and spaces.
            line = line.strip()

            if not line or line.startswith("#"):
                # Count number of empty lines and lines which start with a comment before the actual data.
                skip_rows += 1
            else:
                # Get number of columns from the first real BED entry.
                nbr_columns = len(line.split("\t"))

                # Stop reading the BED file.
                break
    
    if nbr_columns < 5:
        raise ValueError(
            f'Fragments TSV file is incomplete. Five columns are required but "{frag_path}" contains only '
            f"{nbr_columns} columns."
        )
    #
    df = pl.read_csv(
                frag_path,
                has_header=False,
                skip_rows=skip_rows,
                separator="\t",
                use_pyarrow=True,
                new_columns=('Chromosome', 'Start', 'End', 'Barcode','Count'),
            ).with_columns(
                [
                    pl.col("Chromosome").cast(pl.Utf8),
                    pl.col("Start").cast(pl.Int32),
                    pl.col("End").cast(pl.Int32),
                    pl.col("Barcode").cast(pl.Utf8),
                    pl.col("Count").cast(pl.Int32)
                ]
            )
    
    ## construct genome data
    if genome == 'Hg38':
        gen_dataset = pbm.Dataset(name='hsapiens_gene_ensembl',  host='http://www.ensembl.org')
    elif genome == 'Hg19':
        gen_dataset = pbm.Dataset(name='hsapiens_gene_ensembl',  host='http://grch37.ensembl.org/')
    elif genome == 'mm10':
        gen_dataset = pbm.Dataset(name='mmusculus_gene_ensembl',  host='http://nov2020.archive.ensembl.org/')
    elif genome == 'mm39':
        gen_dataset = pbm.Dataset(name='mmusculus_gene_ensembl',  host='http://www.ensembl.org')

    
    print('- Fragment dataframe constructed. Genome version:',genome)
    
    adata.uns['fragment'] = {'file':df, 'genome': gen_dataset}

    return adata