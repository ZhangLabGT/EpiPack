.. EpiPack documentation master file, created by
   sphinx-quickstart on Thu Nov  9 15:43:01 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EpiPack's documentation!
===================================

``epipackpy`` implements the EpiPack computational framework for single-cell ATAC-seq data integration, reference mapping and cell type annotation.

EpiPack leverages a heterogeneous transfer learning framework that accepts gene activity score matrix as bridge and peak embeddings as source information to integrate multi-batch references to construct the cell atlas and transfer cell labels on the common latent space without peak alignment.

.. image:: _static/overview.svg
   :width: 500
   :alt: Model architecture


To get started with ``epipackpy``, check out the `installation guide <install.rst>`__ and `tutorials <tutorials.rst>`__.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting started

   install
   
.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Documentation

   tutorials
