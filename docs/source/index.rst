.. EpiPack documentation master file, created by
   sphinx-quickstart on Thu Nov  9 15:43:01 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EpiPack's documentation!
===================================

EpiPack is a Python package for end-to-end single-cell ATAC-seq analysis, with core functionalities including reference mapping, cell label transfer, and the detection of out-of-reference (OOR) cell types and states under disease or perturbation conditions.

The package consists of three main components:

- ``PEIVI`` and ``PEIVI mapping`` functions for reference construction and query mapping.

- ``epk.classifier`` function for cell label transfer.

- ``Global OOR detector`` and ``local OOR detector`` functions for detecting OOR cells under various conditions.

.. raw:: html

   <div style="text-align: center;">
       <img src="_static/workflow.jpg" alt="Model architecture" width="80%">
   </div>


To get started with ``epipackpy``, check out the `installation guide <install.rst>`__ and `tutorials <tutorials.rst>`__.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting started

   install
   
.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   clustering_tutorial.ipynb
