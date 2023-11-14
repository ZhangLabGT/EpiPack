## EpiPack
EpiPack v 0.1.0dev6

### Description
EpiPack is single-cell ATAC-seq atlas construction (integration) and cell type annotation tool. Leveraging its heterogeneous domain adaptation framework, EpiPack is able to
* Integrate multi-source scATAC-seq datasets without aligned peak set
* reference mapping of query datasets
* cell type annotation including OOR detection

<img src = "figures/overview.png" width = 600ptx>

### Dependency
```
    python >= 3.8
    pytorch >= 1.11.0
    sklearn >= 0.22.1
    numpy >= 1.21.6
    pandas >= 1.3.5
```
Note: For pytorch installation, we recommend users go to the pytorch portal to download based on their CUDA version.

### Installation
The package has been uploaded to PyPI. Users can download the package and dependent packages by:
```
    pip install epipackpy
```

### Tutorial

Please checkout the documentations and tutorials at [epipack.readthedocs.io](https://epipack.readthedocs.io/).
