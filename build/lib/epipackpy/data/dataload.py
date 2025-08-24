import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import sklearn.preprocessing as sp

import polars as pl
import gzip
#import pybiomart as pbm
from os import PathLike
from typing import Literal
import seaborn as sns
import matplotlib.pyplot as plt

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
                      genome:Literal ['Hg38','Hg19','mm39', 'mm10'] = "Hg38",
                      remove_scaffold: bool = True,
                      frag_size_dist: bool = True,
                      max_size: int = 800,
                      add_key: bool = True):
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
    
    if 'fragment' in adata.uns_keys():
        print("- Fragment file already exists. Return without modifying.")
    
    else:

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
            FLT = "chr1|chr2|chr3|chr4|chr5|chr6|chr7|chr8|chr9|chr10|chr11|chr12|chr13|chr14|chr15|chr16|chr17|chr18|chr19|chr20|chr21|chr22|chrX|chrY"
        #    gen_dataset = pbm.Dataset(name='hsapiens_gene_ensembl',  host='http://www.ensembl.org')
        elif genome == 'Hg19':
            FLT = "chr1|chr2|chr3|chr4|chr5|chr6|chr7|chr8|chr9|chr10|chr11|chr12|chr13|chr14|chr15|chr16|chr17|chr18|chr19|chr20|chr21|chr22|chrX|chrY"
        #    gen_dataset = pbm.Dataset(name='hsapiens_gene_ensembl',  host='http://grch37.ensembl.org/')
        elif genome == 'mm10':
            FLT = "chr1|chr2|chr3|chr4|chr5|chr6|chr7|chr8|chr9|chr10|chr11|chr12|chr13|chr14|chr15|chr16|chr17|chr18|chr19|chrX|chrY"
        #    gen_dataset = pbm.Dataset(name='mmusculus_gene_ensembl',  host='http://nov2020.archive.ensembl.org/')
        elif genome == 'mm39':
            FLT = "chr1|chr2|chr3|chr4|chr5|chr6|chr7|chr8|chr9|chr10|chr11|chr12|chr13|chr14|chr15|chr16|chr17|chr18|chr19|chrX|chrY"
        #    gen_dataset = pbm.Dataset(name='mmusculus_gene_ensembl',  host='http://www.ensembl.org')

        if remove_scaffold:
            filter = df['Chromosome'].str.contains(FLT)
            df = df.filter(filter)

        adata.uns['fragment'] = {'file':df, 'path': frag_path, 'genome': genome}
    
        print('- Fragment dataframe constructed. Genome version:',genome)

        # add fragment distribution histogram
        if frag_size_dist:
            print('- Visualizing fragment size distribution...')
            df_ = df.to_pandas(use_pyarrow_extension_array=True)
            df_["Width"] = abs(df_["End"].values - df_["Start"].values)

            frag_width_dict = (
                    df_.groupby(["Width"])
                    .size()
                    .to_frame(name="Count")
                    .rename_axis(None)
                    .reset_index()
                    .rename(columns={"index": "Width"})
                )
            
            sns.lineplot(data=frag_width_dict[frag_width_dict['Width'] < max_size], x='Width', y='Count')
            plt.xlabel("Size of Fragments (bp)")
            plt.ylabel("Count")
            
            if add_key:
                adata.uns['fragment']['frag_dist'] = frag_width_dict

            print("- Done!")
            plt.show()
            

    return adata