# Author: Pierre Ablin <pierre.ablin@ens.fr>
#
# License: MIT
import torch


def linear_stein_kernel(x, y, score_x, score_y, return_kernel=False):
    """Compute the linear Stein kernel between x and y


    Parameters
    ----------
    x : torch.tensor, shape (n, p)
        Input particles
    y : torch.tensor, shape (n, p)
        Input particles
    score_x : torch.tensor, shape (n, p)
        The score of x
    score_y : torch.tensor, shape (n, p)
        The score of y
    return_kernel : bool
        whether the original kernel k(xi, yj) should also be returned

    Return
    ------
    stein_kernel : torch.tensor, shape (n, n)
        The linear Stein kernel
    kernel : torch.tensor, shape (n, n)
        The base kernel, only returned id return_kernel is True
    """
    n, d = x.shape
    kernel = x @ y.t()
    stein_kernel = (
        score_x @ score_y.t() * kernel + score_x @ x.t() + score_y @ y.t() + d
    )
    if return_kernel:
        return stein_kernel, kernel
    return stein_kernel


def gaussian_stein_kernel(
    x, y, scores_x, scores_y, sigma, return_kernel=False
):
    """Compute the Gaussian Stein kernel between x and y


    Parameters
    ----------
    x : torch.tensor, shape (n, p)
        Input particles
    y : torch.tensor, shape (n, p)
        Input particles
    score_x : torch.tensor, shape (n, p)
        The score of x
    score_y : torch.tensor, shape (n, p)
        The score of y
    sigma : float
        Bandwidth
    return_kernel : bool
        whether the original kernel k(xi, yj) should also be returned

    Return
    ------
    stein_kernel : torch.tensor, shape (n, n)
        The linear Stein kernel
    kernel : torch.tensor, shape (n, n)
        The base kernel, only returned id return_kernel is True
    """
    _, p = x.shape
    d = x[:, None, :] - y[None, :, :]
    dists = (d ** 2).sum(axis=-1)
    k = torch.exp(-dists / sigma / 2)
    scalars = scores_x.mm(scores_y.T)
    scores_diffs = scores_x[:, None, :] - scores_y[None, :, :]
    diffs = (d * scores_diffs).sum(axis=-1)
    der2 = p - dists / sigma
    stein_kernel = k * (scalars + diffs / sigma + der2 / sigma)
    if return_kernel:
        return stein_kernel, k
    return stein_kernel


def gaussian_stein_kernel_single(x, score_x, sigma, return_kernel=False):
    """Compute the Gaussian Stein kernel between x and x


    Parameters
    ----------
    x : torch.tensor, shape (n, p)
        Input particles
    score_x : torch.tensor, shape (n, p)
        The score of x
    sigma : float
        Bandwidth
    return_kernel : bool
        whether the original kernel k(xi, xj) should also be returned

    Return
    ------
    stein_kernel : torch.tensor, shape (n, n)
        The linear Stein kernel
    kernel : torch.tensor, shape (n, n)
        The base kernel, only returned id return_kernel is True
    """
    _, p = x.shape
    # Gaussian kernel:
    norms = (x ** 2).sum(-1)
    dists = -2 * x @ x.t() + norms[:, None] + norms[None, :]
    k = (-dists / 2 / sigma).exp()

    # Dot products:
    diffs = (x * score_x).sum(-1, keepdim=True) - (x @ score_x.t())
    diffs = diffs + diffs.t()
    scalars = score_x.mm(score_x.t())
    der2 = p - dists / sigma
    stein_kernel = k * (scalars + diffs / sigma + der2 / sigma)
    if return_kernel:
        return stein_kernel, k
    return stein_kernel


def imq_kernel(x, y, score_x, score_y, g=1, beta=0.5, return_kernel=False):
    """Compute the IMQ Stein kernel between x and y


    Parameters
    ----------
    x : torch.tensor, shape (n, p)
        Input particles
    y : torch.tensor, shape (n, p)
        Input particles
    score_x : torch.tensor, shape (n, p)
        The score of x
    score_y : torch.tensor, shape (n, p)
        The score of y
    g : float
        Bandwidth
    beta : float
        Power of the kernel
    return_kernel : bool
        whether the original kernel k(xi, yj) should also be returned

    Return
    ------
    stein_kernel : torch.tensor, shape (n, n)
        The linear Stein kernel
    kernel : torch.tensor, shape (n, n)
        The base kernel, only returned id return_kernel is True
    """
    _, p = x.shape
    d = x[:, None, :] - y[None, :, :]
    dists = (d ** 2).sum(axis=-1)
    res = 1 + g * dists
    kxy = res ** (-beta)
    scores_d = score_x[:, None, :] - score_y[None, :, :]
    temp = d * scores_d
    dkxy = 2 * beta * g * (res) ** (-beta - 1) * temp.sum(axis=-1)
    d2kxy = 2 * (
        beta * g * (res) ** (-beta - 1) * p
        - 2 * beta * (beta + 1) * g ** 2 * dists * res ** (-beta - 2)
    )
    k_pi = score_x.mm(score_y.T) * kxy + dkxy + d2kxy
    if return_kernel:
        return k_pi, kxy
    return k_pi


def svgd_direction(x, scores_x, sigma):
    """Compute the SVGD direction at x with a Gaussian kernel


    Parameters
    ----------
    x : torch.tensor, shape (n, p)
        Input particles
    score_x : torch.tensor, shape (n, p)
        The score of x
    sigma : float
        Bandwidth

    Return
    ------
    direction : torch.tensor, shape (n, p)
        the SVDG direction
    """
    _, p = x.shape
    d = x[:, None, :] - x[None, :, :]
    dists = (d ** 2).sum(axis=-1)
    k = torch.exp(-dists / sigma / 2)
    der = (d * k[:, :, None]).sum(axis=0) / sigma
    return k.mm(scores_x) - der
