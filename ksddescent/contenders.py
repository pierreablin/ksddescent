# Author: Pierre Ablin <pierre.ablin@ens.fr>
#
# License: MIT
import numpy as np
import torch
from time import time
from scipy.optimize import fmin_l_bfgs_b


def svgd(x0, score, step, max_iter=1000, bw=1, tol=1e-5, verbose=False,
         store=False, backend='auto'):
    """Stein Variational Gradient Descent

    Sample by optimization with the
    Stein Variational Gradient Descent with a Gaussian
    kernel.

    Parameters
    ----------
    x0 : torch.tensor or numpy.ndarray, size n_samples x n_features
        initial positions

    score : callable
        function that computes the score. Must be compatible with the
        backend.

    step : float
        step size

    max_iter : int
        max numer of iters

    bw : float
        bandwidth of the Stein kernel

    tol : float
        tolerance for stopping the algorithm. The
        algorithm stops when the norm of the SVDG direction
        is below tol.

    store : bool
        whether to store the iterates

    verbose : bool
        whether to print the current loss

    backend : str, 'auto', 'numpy' or 'torch'
        Chooses the backend for the algorithm: numpy or torch.
        If `numpy`, `x0` must be a numpy array and `score` must map
        numpy array to numpy array.
        If `torch`, `x0` must be a torch tensor and `score` must map
        torch tensor to torch tensor.
        If `auto`, the backend is infered from the type of `x0`.

    Returns
    -------
    x: torch.tensor or np.ndarray
        The final positions

    References
    ----------
    Q. Liu, D. Wang. Stein variational gradient descent: A general
    purpose Bayesian inference algorithm, Advances In Neural
    Information Processing Systems, 2370-2378
    """
    x_type = type(x0)
    if backend == 'auto':
        if x_type is np.ndarray:
            backend = 'numpy'
        elif x_type is torch.Tensor:
            backend = 'torch'
    if x_type not in [torch.Tensor, np.ndarray]:
        raise TypeError('x0 must be either numpy.ndarray or torch.Tensor '
                        'got {}'.format(x_type))
    if backend not in ['torch', 'numpy', 'auto']:
        raise ValueError('backend must be either numpy or torch, '
                         'got {}'.format(backend))
    if backend == 'torch' and x_type is np.ndarray:
        raise TypeError('Wrong backend')
    if backend == 'numpy' and x_type is torch.Tensor:
        raise TypeError('Wrong backend')
    if backend == 'torch':
        x = x0.detach().clone()
    else:
        x = np.copy(x0)
    n_samples, n_features = x.shape
    if store:
        storage = []
        timer = []
        t0 = time()
    for i in range(max_iter):
        if store:
            if backend == 'torch':
                storage.append(x.clone())
            else:
                storage.append(x.copy())
            timer.append(time() - t0)
        d = (x[:, None, :] - x[None, :, :])
        dists = (d ** 2).sum(axis=-1)

        if backend == 'torch':
            k = torch.exp(- dists / bw / 2)
        else:
            k = np.exp(- dists / bw / 2)
        k_der = d * k[:, :, None] / bw
        scores_x = score(x)

        if backend == 'torch':
            ks = k.mm(scores_x)
        else:
            ks = k.dot(scores_x)
        kd = k_der.sum(axis=0)
        direction = (ks - kd) / n_samples

        criterion = (direction ** 2).sum()
        if criterion < tol ** 2:
            break

        x += step * direction
        if verbose and i % 100 == 0:
            print(i, criterion)
    if store:
        return x, storage, timer
    return x


def gaussian_kernel(x, y, sigma):
    d = (x[:, None, :] - y[None, :, :])
    dists = (d ** 2).sum(axis=-1)
    return torch.exp(- dists / sigma / 2)


def mmd_lbfgs(x0, target_samples, bw=1, max_iter=10000, tol=1e-5,
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

    References
    ----------
    M.Arbel, A.Korba, A.Salim, A.Gretton. Maximum mean discrepancy
    gradient flow, Neurips, 2020.
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
