import numpy as np
import matplotlib.pyplot as plt

def fft(x):
    x = np.asarray(x, dtype=np.complex128)
    N = x.size

    # Base case
    if N == 1:
        return x

    # Must be power of 2
    if N % 2 != 0:
        raise ValueError("FFT manual requires input length to be a power of 2")

    # Recursive Cooley–Tukey
    X_even = fft(x[0::2])
    X_odd  = fft(x[1::2])

    factor = np.exp(-2j * np.pi * np.arange(N) / N)

    X = np.zeros(N, dtype=np.complex128)
    half = N // 2
    X[:half] = X_even + factor[:half] * X_odd
    X[half:] = X_even - factor[:half] * X_odd

    return X

def rfft(x):
    X = fft(x)
    return X[:len(x)//2 + 1]

def stft(x, fs, win_size, hop_size, window='hann'):

    N = len(x)

    if window == 'hann':
        w = np.hanning(win_size)
    else:
        w = np.ones(win_size)

    # jumlah frame
    frames = int(np.floor((N - win_size) / hop_size)) + 1

    n_freqs = win_size // 2 + 1
    freqs = np.linspace(0, fs/2, n_freqs)
    times = (np.arange(frames) * hop_size) / fs

    stft_matrix = np.zeros((n_freqs, frames), dtype=np.complex128)

    for i in range(frames):
        start = i * hop_size
        frame = x[start:start+win_size] * w

        X = rfft(frame)
        stft_matrix[:, i] = X

    return stft_matrix, freqs, times


def plot_stft_spectrogram(stft_matrix, freqs, times, cmap='jet', vmax=None, title="STFT Manual FFT"):
    mag = np.abs(stft_matrix)
    if vmax is None:
        vmax = mag.max() * 0.9

    plt.figure(figsize=(14,6))
    plt.imshow(mag,
               extent=[times[0], times[-1], freqs[0], freqs[-1]],
               origin='lower',
               aspect='auto',
               cmap=cmap,
               vmax=vmax)
    plt.colorbar(label='Magnitude')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
