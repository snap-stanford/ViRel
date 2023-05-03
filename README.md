# ViRel: Unsupervised Visual Relations Discovery with Graph-level Analogy

Abstract:

We introduce Visual Relations with graph-level analogy (ViRel), a method for unsupervised discovery and learning of visual relations with graph-level analogy. In a grid-world based dataset that test visual relation reasoning, it achieves above 95% accuracy in unsupervised relation classification, discovers the relation graph structure for most tasks, and further generalizes to unseen tasks with more complicated relational structures.


Installation
==========================================================================
First clone the directory. Then run the following command to initialize the submodules:

.. code:: bash

   git submodule init; git submodule update

This repository also has the following dependencies:

- `PyTorch <https://pytorch.org/>`_ >= 1.4.0
- `pytorch-geometric <https://github.com/rusty1s/pytorch_geometric>`__
- networkx