CausalDisco
===========

`CausalDisco <https://pypi.org/project/CausalDisco/>`_ is a Python package providing baseline algorithms and analytics tools for Causal Discovery. It is distributed under the open source `3-clause BSD license
<https://github.com/CausalDisco/CausalDisco/blob/main/LICENSE>`_.
The source code can be found on https://github.com/CausalDisco/CausalDisco/.
If you publish work using CausalDisco, please consider citing the publications

- `Beware of the Simulated DAG! <https://proceedings.neurips.cc/paper_files/paper/2021/file/e987eff4a7c7b7e580d659feb6f60c1a-Supplemental.pdf>`_ 
- `Simple Sorting Criteria Help Find the Causal Order in Additive Noise Models <https://arxiv.org/abs/2303.18211>`_.

.. code-block::

    @article{reisach2021beware,
    title={Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy to Game},
    author={Reisach, Alexander G. and Seiler, Christof and Weichwald, Sebastian},
    journal={Advances in Neural Information Processing Systems},
    volume={34},
    year={2021}
    }

    @article{reisach2023simple,
    title={Simple Sorting Criteria Help Find the Causal Order in Additive Noise Models},
    author={Reisach, Alexander G. and Tami, Myriam and Seiler, Christof and Chambaz, Antoine and Weichwald, Sebastian},
    journal={arXiv preprint arXiv:2303.18211},
    year={2023}
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