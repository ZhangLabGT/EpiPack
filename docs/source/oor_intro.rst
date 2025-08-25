Introduction to Global and Local OOR Detection
==================

In the context of scATAC-seq reference mapping, the detection of out-of-reference (OOR) cell types and states is critical for identifying novel biological populations, particularly in disease or perturbed conditions. To better conceptualize this problem, we distinguish two complementary subtypes of OOR phenomena: global OOR and local OOR.

.. raw:: html

   <div style="text-align: center;">
       <img src="_static/oor_framework.jpg" alt="oor_framework" width="80%">
   </div>
   <p style="margin-top:30px;"></p>


**Global OOR**

    Global OOR refers to unseen cell types that are markedly distinct from any in-reference populations.  These cells typically form **discrete clusters** in the joint latent embedding space, separated by clear Euclidean distances from the existing reference clusters.  
    Examples include identifying a completely new immune lineage, such as B cells absent in a T cell–only reference.  In EpiPack, global oor detector is coupled with classifier. For uses who want to find novel cell type when annotating cell labels, please refer to **Global OOR detection** tutorial.


**Local OOR***

    Local OOR captures the emergence of **perturbed or transitional states** that are closely related to in-reference populations but represent a continuous shift along the cellular manifold.  
    Unlike global OOR, local OOR cells do not form isolated clusters but instead manifest as gradual deviations within the local geometry of the embedding, often reflecting disease-associated or activation-induced state changes. Please refer to `Dann et al<https://www.nature.com/articles/s41588-023-01523-7>` for experiment setting tests.
    In EpiPack, we design ``local_oor_detector`` module for this demand. For uses who want to find detect scATAC-seq perturbation populations, please refer to **Local OOR detection** tutorial.


