CausalDisco
===========

`CausalDisco <https://pypi.org/project/CausalDisco/>`_ is a Python package providing baseline algorithms and analytics tools for Causal Discovery. It is distributed under the open source `3-clause BSD license
<https://github.com/CausalDisco/CausalDisco/blob/main/LICENSE>`_.
The source code can be found on https://github.com/CausalDisco/CausalDisco/.
If you publish work using CausalDisco, please consider citing the publications

- `Beware of the Simulated DAG! <https://proceedings.neurips.cc/paper_files/paper/2021/file/e987eff4a7c7b7e580d659feb6f60c1a-Supplemental.pdf>`_ 
- `A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models <https://arxiv.org/abs/2303.18211>`_.

.. code-block::

    @inproceedings{reisach2023scale,
        title = {{A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models}},
        author = {Alexander G. Reisach and Myriam Tami and Christof Seiler and Antoine Chambaz and Sebastian Weichwald},
        booktitle = {{Advances in Neural Information Processing Systems 36 (NeurIPS)}},
        year = {2023},
        doi = {10.48550/arXiv.2303.18211},
    }

    @inproceedings{reisach2021beware,
        title = {{Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy to Game}},
        author = {Alexander G. Reisach and Christof Seiler and Sebastian Weichwald},
        booktitle = {{Advances in Neural Information Processing Systems 34 (NeurIPS)}},
        year = {2021},
        doi = {10.48550/arXiv.2102.13647},
    }


Installation
------------

.. code-block:: bash

    $ pip install CausalDisco

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   
   simple-example.rst
   api-reference.rst
