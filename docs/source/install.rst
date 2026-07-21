Installation guide
==================

************
Main package
************

The ``epipackpy`` package can be installed via pip using the following commands:

.. code-block:: bash

    pip install epipackpy

.. note::
    For GPU-based Pytorch installation, we recommend installing the ``torch`` package on
    `pytorch portal <https://pytorch.org/get-started/locally/>`__
    after checking the CUDA version by ``nvidia-smi``.

For quick environment setup, please download ``epipack.yml`` to retrieve the conda envrionment. ``torch`` package is not included, please check your GPU environment first and install the package from torch website.

If the ``epipack.yml`` is used, please use the following commands:

.. code-block:: bash

    conda env create -f epipack_clean.yml
    conda activate epipack_clean

    pip install --no-deps epipackpy

Now you are all set. Refer to Tutorials for how to use the ``epipackpy`` package.
