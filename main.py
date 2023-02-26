from functools import partial
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Rho:

    def __init__(self, scale=0.1):
        self.rho = 1
        self.scale = scale
        self.randomize()

    def get(self):
        return self.rho

    def randomize(self):
        self.rho = np.random.rand() * self.scale


class Lamda:

    def __init__(self, lamda0):
        self.lamda = lamda0


def fft(x):
    """fourier transform"""
    return np.fft.fft2(x)


def ifft(x):
    """inverse fourier transform"""
    return np.fft.ifft2(x)


def proj_m(z, y):
    """projection operator onto M"""
    fft_z = fft(z)
    return ifft(np.sqrt(y) * (fft_z / np.abs(fft_z)))


def proj_s(z, n):
    """projection operator onto S"""
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
    """the operator A"""
    mask = np.ones_like(z, dtype=bool)
    mask[:n, :n] = False
    return z[mask].flatten()


def a_star(lamda, n):
    """
    the operator A* (adjoint)
    """
    m = int(np.sqrt(lamda.shape[0] + n**2))
    res = np.zeros((m, m), dtype=lamda.dtype)
    idxs = a(np.arange(m**2).reshape((m, m)), n)
    res[np.unravel_index(idxs, (m, m))] = np.conj(lamda)

    return res


def objective_and_grad(z, y, n, rho: Rho, lamda: Lamda):
    """

    Args:
        z: m^2 vector
        y: mxm matrix
        n:
        rho:

    Returns:
        a tuple of the objective and the gradient
    """
    r = rho.get()
    lamda = lamda.lamda

    m = y.shape[0]
    z = real_to_complex(z).reshape((m, m))

    proj_m_z = proj_m(z, y)
    z_minus_proj_m_z = z - proj_m_z
    a_z = a(z, n)

    objective = 0.5 * np.linalg.norm(z_minus_proj_m_z, ord="fro") ** 2 + \
                (r / 2) * np.linalg.norm(a_z + (lamda / r), ord=2) ** 2

    grad = 0.5 * (z_minus_proj_m_z + r * a_star(a_z, n) + a_star(lamda, n))

    return objective, complex_to_real(grad.flatten())


iter = 0


def iteration_callback(z, y, rho, lamda: Lamda, z_0_clean, tol=1e-6):
    global iter
    iter += 1

    rho.randomize()

    z = real_to_complex(z)
    m = int(np.sqrt(z.shape[0]))
    z = z.reshape((m, m))

    # save progress to results folder
    if iter < 10 or iter % 10 == 0:
        save_result(z, iter)

    err = np.linalg.norm(z - proj_m(z, y), ord='fro')

    mse = np.linalg.norm(z_0_clean - z, ord='fro')**2 / (m**2)
    print(f"iter: {iter:>4} | err: {err:.2e} | mse: {mse:.5e} | rho: {rho.get():.2e}")

    return err < tol


def save_result(z, iter):
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(np.abs(z), cmap='gray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    plt.title(f"iter: {iter:>4}")
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig(f"results/iter_{iter:04d}.png")
    plt.close(fig)


def real_to_complex(z):  # real vector of length 2n -> complex of length n
    return z[:len(z) // 2] + 1j * z[len(z) // 2:]


def complex_to_real(z):  # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))


def solve_phase_retrieval(x, rho_scale=0.1, max_iter=int(1e3)):
    global iter

    n = x.shape[0]
    m = 2 * n

    fft_x = fft(pad(x, m))

    y = np.abs(fft_x)**2

    # set noise_scale to 1 to start with random noise, or to 0 to start with the source image.
    # any value inbetween will mix the two proportionaliy.
    noise_scale = 1
    z_0_clean = pad(x, m)
    phase = np.exp(((np.random.rand(m, m) * 2 * np.pi * noise_scale) + np.angle(fft_x) * (1 - noise_scale)) * 1j)
    z_0 = ifft(np.sqrt(y) * phase)

    save_result(z_0, iter=0)

    # the minimizer works only for real numbers so it is required to convert the complex vector
    # to a real vector that is twice as long.
    # this vector will be converted later back to a complex vector for calculation and for analyzing it.
    x_0 = complex_to_real(z_0.flatten())

    rho = Rho(rho_scale)

    # different option for lamda:
    lamda0 = np.ones(m**2 - n**2) * 1
    # lamda0 = np.random.randn(m ** 2 - n ** 2) * 1
    # lamda0 = np.random.rand(m**2 - n**2) * 1 + np.random.rand(m**2 - n**2) * 1j
    # lamda0 = (np.random.rand(m**2 - n**2) - 0.5) * 1 + (np.random.rand(m**2 - n**2) - 0.5) * 1j

    lamda = Lamda(lamda0)

    partial_callback = partial(iteration_callback, y=y, lamda=lamda, rho=rho, z_0_clean=z_0_clean)

    # the minimize function doesn't let us to disable the internal stop condition
    # so in order to run the desired amount of iterations we have to restart the minimization
    # using the result of the recent minimization as the starting point of the next
    iter = 0
    while iter < max_iter:
        opts = {
            'maxiter': max_iter - iter,
            'disp': False,
            'gtol': 0,
            'ftol': 0,
            'maxls': 50,
        }
        res = minimize(fun=objective_and_grad,
                       x0=x_0,
                       jac=True,
                       args=(y, n, rho, lamda),
                       method='L-BFGS-B',
                       callback=partial_callback,
                       bounds=None,
                       options=opts)
        x_0 = res.x

    x_hat = real_to_complex(res.x).reshape((m, m))[:n, :n]
    print(f"SUCCESS: {np.allclose(x, x_hat, atol=0.05)}")

    return x_hat


def main():
    np.random.seed(1)

    size_px = 50
    resampling = Image.Resampling.BICUBIC

    # load image
    im = Image.open(os.path.join("images", "dancer.jpg")).convert('L')
    im_resized = im.resize((size_px, size_px), resample=resampling)
    x = np.asarray(im_resized) / 255.0

    # solve PR
    x_hat = solve_phase_retrieval(x)
    x_hat_real = np.clip(np.real(x_hat), 0, 1)

    diff = np.abs(x - x_hat_real)

    # plot original image and result
    fig, axs = plt.subplots(1, 3, width_ratios=(1, 1, 1.06), figsize=(12, 4))

    axs[0].imshow(x, cmap="gray")
    axs[0].set_title("Original")

    axs[1].imshow(x_hat_real, cmap="gray")
    axs[1].set_title("Recovered")

    im_diff = axs[2].imshow(diff, cmap="plasma")
    axs[2].set_title("Difference")

    # aAdd colorbar
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_diff, cax=cax)

    plt.show()


if __name__ == "__main__":
    main()
