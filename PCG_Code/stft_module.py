"""
STFT Module - Short-Time Fourier Transform Implementation
Manual implementation without external FFT libraries
"""

import numpy as np


def _get_manual_window(window_name, N):
    """Generate window function"""
    n = np.arange(N, dtype=float)
    if window_name is None:
        return np.ones(N, dtype=float)
    if isinstance(window_name, str):
        w_name = window_name.lower()
        if 'hann' in w_name:
            if N == 1: return np.ones(1)
            return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (N - 1))
    return np.ones(N, dtype=float)


def _is_power_of_two(n):
    """Check if number is power of 2"""
    return (n != 0) and ((n & (n - 1)) == 0)


def _manual_fft_recursive(x):
    """Recursive FFT implementation"""
    N = x.shape[0]
    if N <= 1:
        return x
    even = _manual_fft_recursive(x[0::2])
    odd  = _manual_fft_recursive(x[1::2])
    k = np.arange(N // 2)
    factor = np.exp(-2j * np.pi * k / N)
    return np.concatenate([even + factor * odd, even - factor * odd])


def _manual_dft_slow(x):
    """Slow DFT for non-power-of-2 sizes"""
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def _manual_rfft(segment, nfft):
    """Real FFT implementation"""
    if len(segment) < nfft:
        padded = np.zeros(nfft, dtype=float)
        padded[:len(segment)] = segment
        x_in = padded
    else:
        x_in = segment[:nfft]
    if _is_power_of_two(nfft):
        X_full = _manual_fft_recursive(x_in)
    else:
        X_full = _manual_dft_slow(x_in)
    return X_full[:nfft//2 + 1]


def _manual_rfftfreq(nfft, d=1.0):
    """Generate frequency bins for RFFT"""
    val = 1.0 / (nfft * d)
    N = nfft // 2 + 1
    return np.arange(0, N, dtype=float) * val


def compute_stft(x, fs=2000, nperseg=None, noverlap=None, nfft=None,
                 window='hann', scaling='density', mode='magnitude'):
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if N == 0:
        return np.zeros((0, 0)), np.zeros((0,)), np.zeros((0,)), "STFT"
    if nperseg is None:
        nperseg = min(256, max(64, max(1, N // 8)))
    if nfft is None:
        nfft = max(256, nperseg)
    if noverlap is None:
        noverlap = int(nperseg // 2)
    nperseg, nfft, noverlap = int(max(1, nperseg)), int(max(1, nfft)), int(max(0, noverlap))
    hop = nperseg - noverlap
    if hop <= 0:
        hop = 1
    win = _get_manual_window(window, nperseg)
    if N <= nperseg:
        n_frames = 1
    else:
        n_frames = int(np.ceil((N - noverlap) / float(hop)))
    pad_len = (n_frames - 1) * hop + nperseg
    if pad_len > N:
        x_padded = np.concatenate([x, np.zeros(pad_len - N, dtype=float)])
    else:
        x_padded = x[:pad_len]
    frames, times = [], []
    for i in range(0, pad_len - nperseg + 1, hop):
        seg = x_padded[i : i + nperseg] * win
        X_complex = _manual_rfft(seg, nfft)
        mag = np.sqrt(X_complex.real**2 + X_complex.imag**2)
        frames.append(mag)
        times.append((i + (nperseg / 2.0)) / float(fs))
    if len(frames) == 0:
        freqs = _manual_rfftfreq(nfft, d=1.0/fs)
        return np.zeros((freqs.size, 0)), freqs, np.array([], dtype=float), "STFT"
    S = np.column_stack(frames)
    freqs = _manual_rfftfreq(nfft, d=1.0/fs)
    return S, freqs, np.asarray(times, dtype=float), "STFT"