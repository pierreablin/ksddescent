import numpy as np
from numpy.testing import assert_raises
import torch
import pytest
from scipy.stats import kurtosis

from ksddescent import svgd


@pytest.mark.parametrize("backend", ["numpy", "torch", "auto", "choucroute"])
@pytest.mark.parametrize("input_type", ["numpy", "torch", "str"])
@pytest.mark.parametrize("n, p", [(1, 1), (1, 10), (10, 1), (10, 3)])
def test_output(backend, input_type, n, p):
    def score(x):
        return -(x ** 3)

    max_iter = 2
    if input_type == "numpy":
        x = np.random.randn(n, p)
    elif input_type == "torch":
        x = torch.randn(n, p)
    else:
        x = "couscous"
    step = 1.0
    if (
        input_type == "str"
        or (backend == "numpy" and input_type == "torch")
        or (backend == "torch" and input_type == "numpy")
    ):
        with assert_raises(TypeError):
            output = svgd(x, score, step, max_iter=max_iter, backend=backend)
        return
    if backend == "choucroute":
        with assert_raises(ValueError):
            output = svgd(x, score, step, max_iter=max_iter, backend=backend)
        return

    output = svgd(x, score, step, max_iter=max_iter, backend=backend)
    assert output.shape == (n, p)
    if backend == "torch":
        assert output.requires_grad is False


def test_gaussian():
    torch.manual_seed(0)

    def score(x):
        return -x

    max_iter = 1000
    n, p = 100, 1
    x = torch.rand(n, p)
    step = 1.0
    output = svgd(x, score, step, max_iter=max_iter)
    assert np.abs(kurtosis(output[:, 0])) < 0.1
