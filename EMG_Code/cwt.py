import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def morlet(t, s, fs, w0=6.0):

    ts = t / s
    psi = (np.pi ** (-0.25)) * np.exp(1j * w0 * ts) * np.exp(-0.5 * ts * ts)
    return psi / np.sqrt(s)


def freq2scale(freq, fs, w0=6.0):

    dt = 1.0 / fs
    s = w0 / (2 * np.pi * freq * dt)  
    return s * dt


def cwt_morlet(x, fs, freqs, w0=6.0, nv=3.0):

    N = len(x)
    dt = 1.0 / fs
    coeffs = np.zeros((len(freqs), N), dtype=np.complex128)

    for i, f in enumerate(freqs):
        
        scale_s = freq2scale(f, fs, w0=w0)  
        half_width_sec = nv * scale_s
        half_width_samples = max(1, int(np.ceil(half_width_sec * fs)))
        t = np.arange(-half_width_samples, half_width_samples + 1) / fs  

        psi = morlet(t, scale_s, fs, w0=w0)
        conv = np.convolve(x, psi[::-1].conjugate(), mode='same')

        coeffs[i, :] = conv

    return coeffs, freqs


def plot_scalograms_two_signals(coeffs_vl, coeffs_gl, freqs, fs, t0=0.0, cmap='jet', vmax=None):

    N_1 = coeffs_vl.shape[1]
    T_1 = N_1 / fs
    t_1 = np.linspace(0, T_1, N_1)\
    
    N_2 = coeffs_vl.shape[1]
    T_2 = N_2 / fs
    t_2 = np.linspace(0, T_1, N_1)


    mag_vl = np.abs(coeffs_vl)
    mag_gl = np.abs(coeffs_gl)

    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    im1 = axs[0].imshow(mag_vl, extent=[t_1[0], t_1[-1], 1/freqs[-1], freqs[0]],
                        aspect='auto', cmap=cmap, vmax=vmax)
    axs[0].set_title("CWT Morlet Scalogram - VL")
    axs[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im1, ax=axs[0], orientation='vertical', label='|CWT|')

    im2 = axs[1].imshow(mag_gl, extent=[t_2[0], t_2[-1], 1/freqs[-1], freqs[0]],
                        aspect='auto', cmap=cmap, vmax=vmax)
    axs[1].set_title("CWT Morlet Scalogram - GL")
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].set_xlabel("Time (s)")
    fig.colorbar(im2, ax=axs[1], orientation='vertical', label='|CWT|')

    plt.tight_layout()
    plt.show()

def plot_cwt_3d(coeffs, freqs, fs, title="3D CWT Morlet", stride=2):

    N = coeffs.shape[1]
    T = N / fs
    t = np.linspace(0, T, N)

    mag = np.abs(coeffs)

    T_grid, F_grid = np.meshgrid(t, freqs)

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        T_grid, F_grid, mag,
        rstride=stride, cstride=stride,
        cmap='jet', linewidth=0, antialiased=True
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_zlabel("|CWT| Magnitude")
    ax.set_title(title)

    fig.colorbar(surf, shrink=0.6, aspect=10)
    plt.tight_layout()
    plt.show()
