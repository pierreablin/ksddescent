import numpy as np
import torch
import pytest
from scipy.stats import kurtosis

from ksddescent import svgd


@pytest.mark.parametrize('n, p', [(1, 1), (1, 10), (10, 1), (10, 3)])
def test_output(n, p):
    def score(x):
        return - x ** 3
    max_iter = 1
    x = torch.randn(n, p)
    step = 1.
    output = svgd(x, score, step, max_iter=max_iter)
    assert output.shape == (n, p)
    assert output.requires_grad is False


def test_gaussian():
    torch.manual_seed(0)

    def score(x):
        return - x
    max_iter = 1000
    n, p = 100, 1
    x = torch.rand(n, p)
    step = 1.
    output = svgd(x, score, step, max_iter=max_iter)
    assert np.abs(kurtosis(output[:, 0])) < .1
