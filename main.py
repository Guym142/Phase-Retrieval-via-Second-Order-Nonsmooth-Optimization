from functools import partial
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class Rho:
    def __init__(self, scale=0.1):
        self.rho = None
        self.scale = scale
        self.randomize()

    def get(self):
        return self.rho

    def randomize(self):
        self.rho = np.random.rand() * self.scale


def fft_norm(x):
    """normalized 2d fft"""
    n = x.shape[0]
    return np.fft.fft2(x) / np.sqrt(n)


def ifft_norm(x):
    """normalized 2d invers fft"""
    n = x.shape[0]
    return np.fft.ifft2(x) * np.sqrt(n)


def proj_m(z, y):
    fft_z = fft_norm(z)
    return ifft_norm(np.sqrt(y) * (fft_z / np.abs(fft_z)))


def proj_s(z, n):
    z_proj = np.zeros_like(z)
    z_proj[:n, :n] = z[:n, :n]
    return z_proj


def pad(x, m):
    """pad x to size m x m"""
    n = x.shape[0]
    x_pad = np.zeros((m, m))
    x_pad[:n, :n] = x
    return x_pad


def a(z, n):
    mask = np.ones_like(z, dtype=bool)
    mask[:n, :n] = False
    return z[mask].flatten()


def a_star(gamma, n):
    """

    Args:
        gamma: vector of length m^2 - n^2
        n:

    Returns:

    """
    m = int(np.sqrt(gamma.shape[0] + n ** 2))
    res = np.zeros((m, m), dtype=gamma.dtype)
    idxs = a(np.arange(m ** 2).reshape((m, m)), n)
    res[np.unravel_index(idxs, (m, m))] = gamma

    return res


def g(z, y, n, rho: Rho):
    """

    Args:
        z: m^2 vector
        y: mxm matrix
        n:
        rho:

    Returns:

    """
    rho.randomize()
    r = rho.get()
    m = y.shape[0]
    z = real_to_complex(z)
    z = z.reshape((m, m))

    gamma = 1
    target = (1 / 2) * np.linalg.norm(z - proj_m(z, y), ord="fro") ** 2 + (r / 2) * np.linalg.norm(
        a(z, n) + (gamma / r),
        ord=2) ** 2

    print(f"target: {target:.3f}, rho: {r:.3f}")
    return target


def grad_g(z, y, n, rho: Rho):
    m = y.shape[0]
    z = real_to_complex(z)
    z = z.reshape((m, m))
    r = rho.get()

    gamma = np.ones(m ** 2 - n ** 2)
    grad_z = 1 / 2 * (z - proj_m(z, y) + r * a_star(a(z, n), n) + a_star(gamma, n)).flatten()
    return complex_to_real(grad_z)


def stop_callback(z, n, tol=10e-6):
    z = real_to_complex(z)
    m = int(np.sqrt(z.shape[0]))
    z = z.reshape((m, m))
    return np.linalg.norm(z - proj_s(z, n), ord='fro') < tol


def real_to_complex(z):  # real vector of length 2n -> complex of length n
    return z[:len(z) // 2] + 1j * z[len(z) // 2:]


def complex_to_real(z):  # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))


def main():
    x = np.random.rand(25, 25)  # get image
    n = x.shape[0]
    m = 2 * n - 1
    stop_callback_partial = partial(stop_callback, n=n)

    rho = Rho()

    padded_x = pad(x, m)
    y = np.abs(fft_norm(padded_x)) ** 2
    z_0 = ifft_norm(np.sqrt(y) * np.exp(1j * np.random.rand(m, m) * 2 * np.pi))

    opts = {'maxiter': int(1e4),
            'disp': True,
            'gtol': -np.inf,
            'ftol': -np.inf,
            }
    res = minimize(fun=g,
                   x0=complex_to_real(z_0.flatten()),
                   jac=grad_g,
                   args=(y, n, rho),
                   method='L-BFGS-B',
                   callback=stop_callback_partial,
                   options=opts, tol=-np.inf)

    x_hat = real_to_complex(res.x).reshape((m, m))[:n, :n]
    print(res)
    # assert np.allclose(x, x_hat, atol=1e-3)

    # n = 4
    # x = np.arange(n ** 2).reshape((n, n))
    # print(x)
    # print(a(x, 2))
    # print(a_star(a(x, 2), 2))


if __name__ == "__main__":
    main()
