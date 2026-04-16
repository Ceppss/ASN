"""
CWT Module - Continuous Wavelet Transform Implementation
Manual implementation using Morlet wavelet
"""

import numpy as np


def _get_morlet_kernel(M, w0=5.0):
    """Generate Morlet wavelet kernel"""
    t = np.arange(-M // 2, M // 2)
    sigma = M / 8.0 
    t_scaled = t / sigma
    c = np.pi ** (-0.25)
    envelope = np.exp(-0.5 * t_scaled**2)
    oscillation = np.cos(w0 * t_scaled) + 1j * np.sin(w0 * t_scaled)
    psi = c * envelope * oscillation
    return psi


def _manual_convolution(signal, kernel):
    """Perform manual convolution"""
    return np.convolve(signal, kernel, mode='same')


def _compute_cwt_manual(signal, fs, fmin=20.0, fmax=500.0, n_freqs=120, w0=5.0):
    """Compute CWT manually using Morlet wavelet"""
    signal = np.asarray(signal)
    N = signal.size
    dt = 1.0 / fs
    freqs = np.linspace(fmin, fmax, n_freqs)
    scales = (w0 * fs) / (2.0 * np.pi * freqs)
    cwt_matrix = np.zeros((n_freqs, N), dtype=float)
    
    for i, a in enumerate(scales):
        window_size = int(min(10 * a, N))
        if window_size % 2 == 0: 
            window_size += 1
        kernel = _get_morlet_kernel(window_size, w0=w0)
        norm_factor = 1.0 / np.sqrt(a)
        kernel = kernel * norm_factor
        conv_res = _manual_convolution(signal, kernel)
        power = (conv_res.real ** 2) + (conv_res.imag ** 2)
        cwt_matrix[i, :] = power

    times = np.arange(N) * dt
    maxv = np.nanmax(cwt_matrix) if cwt_matrix.size else 0.0
    if maxv > 0:
        cwt_matrix = cwt_matrix / (maxv + 1e-12)
    
    return cwt_matrix, freqs, times


def compute_cwt_pascal(signal, fs, fmin=20.0, fmax=500.0, n_freqs=120):
    """
    Compute CWT using Pascal-style implementation
    
    Parameters:
    -----------
    signal : array-like
        Input signal
    fs : float
        Sampling frequency
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
    n_freqs : int
        Number of frequency bins
        
    Returns:
    --------
    scalogram : ndarray
        2D array of CWT coefficients
    freqs : ndarray
        Frequency array
    times : ndarray
        Time array
    """
    return _compute_cwt_manual(signal, fs, fmin=fmin, fmax=fmax, n_freqs=n_freqs)