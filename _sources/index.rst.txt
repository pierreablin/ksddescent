Kernel Stein Discrepancy Descent
================================

Sampling by optimization of the Kernel Stein Discrepancy

The paper is available at `https://arxiv.org/abs/2105.09994 <https://arxiv.org/abs/2105.09994>`_.

The code uses Pytorch, and a numpy backend is available for SVGD.

.. image:: https://pierreablin.github.io/figures/ksd_descent.png
    :width: 200
    :alt: ksd_picture


Install
-------

The code is available on pip::

	$ pip install ksddescent


The github repository is at `github.com/pierreablin/ksddescent <https://github.com/pierreablin/ksddescent>`_


Example
-------

The main function is `ksdd_lbfgs`, which uses the fast L-BFGS algorithm to converge quickly.
It takes as input the initial position of the particles, and the score function.
For instance, to samples from a Gaussian (where the score is identity), you can use these simple lines of code:

.. code:: python

   >>> import torch
   >>> from ksddescent import ksdd_lbfgs
   >>> n, p = 50, 2
   >>> x0 = torch.rand(n, p)  # start from uniform distribution
   >>> score = lambda x: x  # simple score function
   >>> x = ksdd_lbfgs(x0, score)  # run the algorithm


Reference
---------

If you use this code in your project, please cite::

    Anna Korba, Pierre-Cyril Aubin-Frankowski, Simon Majewski, Pierre Ablin
    Kernel Stein Discrepancy Descent
    International Conference on Machine Learning, 2021





Bug reports
-----------

Use the `github issue tracker <https://github.com/pierreablin/ksddescent/issues>`_ to report bugs.
