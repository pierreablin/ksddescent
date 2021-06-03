import numpy as np
import torch
import pytest
from scipy.stats import kurtosis

from ksddescent import ksdd_lbfgs, ksdd_gradient


@pytest.mark.parametrize('algo', ['lbfgs', 'gradient'])
@pytest.mark.parametrize('n, p', [(1, 1), (1, 10), (10, 1), (10, 3)])
def test_output(algo, n, p):
    def score(x):
        return - x ** 3
    max_iter = 1
    x = torch.randn(n, p)
    if algo == 'lbfgs':
        output = ksdd_lbfgs(x, score, max_iter=max_iter)
    else:
        step = 1.
        output = ksdd_gradient(x, score, step, max_iter=max_iter)
    assert output.shape == (n, p)
    assert output.requires_grad is False


@pytest.mark.parametrize('algo', ['lbfgs', 'gradient'])
def test_gaussian(algo):
    torch.manual_seed(0)

    def score(x):
        return - x
    max_iter = 300
    n, p = 100, 1
    x = torch.rand(n, p)
    if algo == 'lbfgs':
        output = ksdd_lbfgs(x, score, max_iter=max_iter)
    else:
        step = 40
        output = ksdd_gradient(x, score, step, max_iter=max_iter)
    assert np.abs(kurtosis(output[:, 0])) < .1
