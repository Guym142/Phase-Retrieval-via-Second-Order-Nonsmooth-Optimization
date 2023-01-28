from functools import partial
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from PIL import Image


class Rho:

    def __init__(self, scale=0.1):
        self.rho = None
        self.scale = scale
        self.randomize()

    def get(self):
        return self.rho

    def randomize(self):
        self.rho = np.random.rand() * self.scale


def fft(x):
    """normalized 2d fft"""
    return np.fft.fft2(x)


def ifft(x):
    """normalized 2d invers fft"""
    return np.fft.ifft2(x)


def proj_m(z, y):
    fft_z = fft(z)
    return ifft(np.sqrt(y) * (fft_z / np.abs(fft_z)))


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
    m = int(np.sqrt(gamma.shape[0] + n**2))
    res = np.zeros((m, m), dtype=gamma.dtype)
    idxs = a(np.arange(m**2).reshape((m, m)), n)
    res[np.unravel_index(idxs, (m, m))] = gamma

    return res


def objective_and_grad(z, y, n, rho: Rho):
    """

    Args:
        z: m^2 vector
        y: mxm matrix
        n:
        rho:

    Returns:

    """
    r = rho.get()
    m = y.shape[0]
    z = real_to_complex(z).reshape((m, m))

    gamma = np.ones(m**2 - n**2)
    proj_m_z = proj_m(z, y)
    z_minus_proj_m_z = z - proj_m_z
    a_z = a(z, n)

    objective = 0.5 * np.linalg.norm(z_minus_proj_m_z, ord="fro")**2 + \
        (r / 2) * np.linalg.norm(a_z + (gamma / r), ord=2)**2

    grad = 0.5 * (z_minus_proj_m_z + r * a_star(a_z, n) + a_star(gamma, n))

    return (objective, complex_to_real(grad.flatten()))


iter = 0


def iteration_callback(z, n, rho, tol=1e-6):
    global iter
    iter += 1

    rho.randomize()

    z = real_to_complex(z)
    m = int(np.sqrt(z.shape[0]))
    z = z.reshape((m, m))

    err = np.linalg.norm(z - proj_m(z, n), ord='fro')

    print(f"iter: {iter:>4} | err: {err:.2e}")

    return err < tol


def real_to_complex(z):  # real vector of length 2n -> complex of length n
    return z[:len(z) // 2] + 1j * z[len(z) // 2:]


def complex_to_real(z):  # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))


def solve_phase_retrieval(x, rho_scale=0.1, max_iter=int(1e2)):
    global iter

    n = x.shape[0]
    m = 2 * n - 1

    y = np.abs(fft(pad(x, m)))**2

    z_0 = ifft(np.sqrt(y) * np.exp(1j * np.random.rand(m, m) * 2 * np.pi))
    x_0 = complex_to_real(z_0.flatten())

    rho = Rho(rho_scale)
    partial_callback = partial(iteration_callback, n=n, rho=rho)

    iter = 0
    while iter < max_iter:
        opts = {
            'maxiter': max_iter - iter,
            'disp': False,
            'gtol': 0,
            'ftol': 0,
        }
        res = minimize(fun=objective_and_grad,
                       x0=x_0,
                       jac=True,
                       args=(y, n, rho),
                       method='L-BFGS-B',
                       callback=partial_callback,
                       bounds=None,
                       options=opts)
        x_0 = res.x

    x_hat = real_to_complex(res.x).reshape((m, m))[:n, :n]
    print(f"SUCCESS: {np.allclose(x, x_hat, atol=0.05)}")

    return x_hat


def main():
    size_px = 25
    resampling = Image.Resampling.BICUBIC

    # Load image
    im = Image.open(os.path.join("images", "dancer.jpg")).convert('L')
    im_resized = im.resize((size_px, size_px), resample=resampling)
    x = np.asarray(im_resized) / 255.0

    # solve PR
    x_hat = solve_phase_retrieval(x)

    # plot original image and result
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(x, cmap="gray")
    axs[1].imshow(np.clip(np.real(x_hat), 0, 1), cmap="gray")
    plt.show()

    # n = 4
    # x = np.arange(n ** 2).reshape((n, n))
    # print(x)
    # print(a(x, 2))
    # print(a_star(a(x, 2), 2))


if __name__ == "__main__":
    main()
