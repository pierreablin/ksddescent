# Author: Pierre Ablin <pierre.ablin@ens.fr>
#
# License: MIT
import numpy as np
import torch
from time import time
from scipy.optimize import fmin_l_bfgs_b


def svgd(x0, score, step, n_iter=1000, bw=1, verbose=False,
         store=False):
    """Stein Variational Gradient Descent

    Sample by optimization with the
    Stein Variational Gradient Descent

    Parameters
    ----------
    x0 : torch.tensor, size n_samples x n_features
        initial positions

    score : callable
        function that computes the score

    step : float
        step size

    n_iter : int
        max numer of iters

    bw : float
        bandwidth of the stein kernel

    store : bool
        whether to store the iterates

    verbose: bool
        whether to print the current loss

    Returns
    -------
    x: torch.tensor
        The final positions

    Reference
    ---------
    Q. Liu, D. Wang. Stein variational gradient descent: A general
    purpose Bayesian inference algorithm, Advances In Neural
    Information Processing Systems, 2370-2378
    """
    x = x0.detach().clone()
    n_samples, n_features = x.shape
    if store:
        storage = []
        timer = []
        t0 = time()
    for i in range(n_iter):
        if store:
            storage.append(x.clone())
            timer.append(time() - t0)
        d = (x[:, None, :] - x[None, :, :])
        dists = (d ** 2).sum(axis=-1)
        k = torch.exp(- dists / bw / 2)
        k_der = d * k[:, :, None] / bw
        scores_x = score(x)
        ks = k.mm(scores_x)
        kd = k_der.sum(dim=0)
        direction = (ks - kd) / n_samples
        x += step * direction
        if verbose and i % 100 == 0:
            print(i, torch.norm(direction).item())
    if store:
        return x, storage, timer
    return x


def gaussian_kernel(x, y, sigma):
    d = (x[:, None, :] - y[None, :, :])
    dists = (d ** 2).sum(axis=-1)
    return torch.exp(- dists / sigma / 2)


def mmd_lbfgs(x0, target_samples, bw=1, max_iter=10000, tol=1e-12,
              store=False):
    '''Sampling by optimization of the MMD

    This uses target samples from a base distribution and
    returns new samples by minimizing the maximum mean discrepancy.
    Parameters
    ----------

    x0 : torch.tensor, size n_samples x n_features
        initial positions

    target_samples : torch.tensor, size n_samples x n_features
        Samples from the target distribution

    bw : float
        bandwidth of the stein kernel

    max_iter : int
        max numer of iters

    tol : float
        tolerance for L-BFGS

    Returns
    -------

    x: torch.tensor
        The final positions
    '''
    x = x0.clone().detach().numpy()
    n_samples, p = x.shape
    k_yy = gaussian_kernel(target_samples, target_samples, bw).sum().item()
    if store:
        class callback_store():
            def __init__(self):
                self.t0 = time()
                self.mem = []
                self.timer = []

            def __call__(self, x):
                self.mem.append(np.copy(x))
                self.timer.append(time() - self.t0)

            def get_output(self):
                storage = [torch.tensor(x.reshape(n_samples, p),
                                        dtype=torch.float32)
                           for x in self.mem]
                return storage, self.timer
        callback = callback_store()
    else:
        callback = None

    def loss_and_grad(x_numpy):
        x_numpy = x_numpy.reshape(n_samples, p)
        x = torch.tensor(x_numpy, dtype=torch.float32)
        x.requires_grad = True
        k_xx = gaussian_kernel(x, x, bw).sum()
        k_xy = gaussian_kernel(x, target_samples, bw).sum()
        loss = k_xx - 2 * k_xy + k_yy
        loss.backward()
        grad = x.grad
        return loss.item(), np.float64(grad.numpy().ravel())

    t0 = time()
    x, f, d = fmin_l_bfgs_b(loss_and_grad, x.ravel(), maxiter=max_iter,
                            factr=tol, epsilon=1e-12, pgtol=1e-10,
                            callback=callback)
    print('Took %.2f sec, %d iterations, loss = %.2e' %
          (time() - t0, d['nit'], f))
    output = torch.tensor(x.reshape(n_samples, p), dtype=torch.float32)
    if store:
        storage, timer = callback.get_output()
        return output, storage, timer
    return output
