# Author: Pierre Ablin <pierre.ablin@ens.fr>
#
# License: MIT
import torch
import numpy as np
from .kernels import (imq_kernel, gaussian_stein_kernel_single,
                      linear_stein_kernel)

from scipy.optimize import fmin_l_bfgs_b
from time import time


def ksdd_gradient(x0, score, step, kernel='gaussian', max_iter=1000, bw=1,
                  store=False, verbose=False, clamp=None, beta=0.2):
    '''Kernel Stein Discrepancy descent with gradient descent

    Perform Kernel Stein Discrepancy descent with gradient descent.
    Since it uses gradient descent, a step size must be specified.

    Parameters
    ----------
    x0 : torch.tensor, size n_samples x n_features
        initial positions

    score : callable
        function that computes the score

    step : float
        step size

    max_iter : int
        max numer of iters

    bw : float
        bandwidth of the stein kernel

    stores : None or list of ints
        whether to stores the iterates at the indexes in the list

    verbose: bool
        wether to print the current loss

    clamp:
        if not None, should be a tuple (a, b). The points x are then
        constrained to stay in [a, b]

    Returns
    -------
    x: torch.tensor, size n_samples x n_features
        The final positions

    loss_list : list of floats
        List of the loss values during iterations

    References
    ----------
    A.Korba, P-C. Aubin-Frankowski, S.Majewski, P.Ablin.
    Kernel Stein Discrepancy Descent
    International Conference on Machine Learning, 2021.
    '''
    x = x0.clone().detach()
    n_samples, p = x.shape
    x.requires_grad = True
    if store:
        storage = []
        timer = []
        t0 = time()
    loss_list = []
    n = None
    for i in range(max_iter + 1):
        if store:
            timer.append(time() - t0)
            storage.append(x.clone())
        scores_x = score(x)
        if kernel == 'gaussian':
            K = gaussian_stein_kernel_single(x, scores_x, bw)
        else:
            K = imq_kernel(x, x, scores_x, scores_x, g=bw, beta=beta)
        loss = K.sum() / n_samples ** 2
        loss.backward()
        loss_list.append(loss.item())
        if verbose and i % 100 == 0:
            print(i, loss.item())
        with torch.no_grad():
            x[:] -= step * x.grad
            if n is not None:
                x[:] -= n
            if clamp is not None:
                x = x.clamp(clamp[0], clamp[1])
            x.grad.data.zero_()
        x.requires_grad = True
    x.requires_grad = False
    if store:
        return x, storage, timer
    else:
        return x


def ksdd_lbfgs(x0, score, kernel='gaussian', bw=1.,
               max_iter=10000, tol=1e-12, beta=.5,
               store=False, verbose=False):
    '''Kernel Stein Discrepancy descent with L-BFGS

    Perform Kernel Stein Discrepancy descent with L-BFGS.
    L-BFGS is a fast and robust algorithm, that has no
    critical hyper-parameter.

    Parameters
    ----------
    x0 : torch.tensor, size n_samples x n_features
        initial positions

    score : callable
        function that computes the score

    kernl : 'gaussian' or 'imq'
        which kernel to choose

    max_iter : int
        max numer of iters

    bw : float
        bandwidth of the stein kernel

    tol : float
        stopping criterion for L-BFGS

    store : bool
        whether to stores the iterates

    verbose: bool
        wether to print the current loss

    Returns
    -------
    x: torch.tensor, size n_samples x n_features
        The final positions

    References
    ----------
    A.Korba, P-C. Aubin-Frankowski, S.Majewski, P.Ablin.
    Kernel Stein Discrepancy Descent
    International Conference on Machine Learning, 2021.
    '''
    x = x0.clone().detach().numpy()
    n_samples, p = x.shape
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
        scores_x = score(x)
        if kernel == 'gaussian':
            stein_kernel = gaussian_stein_kernel_single(x, scores_x, bw)
        elif kernel == 'imq':
            stein_kernel = imq_kernel(x, x, scores_x, scores_x, bw, beta=beta)
        else:
            stein_kernel = linear_stein_kernel(x, x, scores_x, scores_x)
        loss = stein_kernel.sum()
        loss.backward()
        grad = x.grad
        return loss.item(), np.float64(grad.numpy().ravel())

    t0 = time()
    x, f, d = fmin_l_bfgs_b(loss_and_grad, x.ravel(), maxiter=max_iter,
                            factr=tol, epsilon=1e-12, pgtol=1e-10,
                            callback=callback)
    if verbose:
        print('Took %.2f sec, %d iterations, loss = %.2e' %
              (time() - t0, d['nit'], f))
    output = torch.tensor(x.reshape(n_samples, p), dtype=torch.float32)
    if store:
        storage, timer = callback.get_output()
        return output, storage, timer
    return output
