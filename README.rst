Kernel Stein Discrepancy Descent
================================

|GHActions|_ |PyPI|_ 

.. |GHActions| image:: https://github.com/pierreablin/ksddescent/workflows/unittests/badge.svg?branch=main&event=push
.. _GHActions: https://github.com/pierreablin/ksddescent/actions


.. |PyPI| image:: https://badge.fury.io/py/ksddescent.svg
.. _PyPI: https://badge.fury.io/py/ksddescent

Sampling by optimization of the Kernel Stein Discrepancy

The paper is available at `arxiv.org/abs/2105.09994 <https://arxiv.org/abs/2105.09994>`_.

The code uses Pytorch, and a numpy backend is available for svgd.


.. image:: https://pierreablin.github.io/figures/ksd_descent_small.png
    :width: 100
    :alt: ksd_picture


Install
-------

The code is available on pip::

	$ pip install ksddescent


Documentation
-------------

The documentation is at `pierreablin.github.io/ksddescent/ <https://pierreablin.github.io/ksddescent/>`_.

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
