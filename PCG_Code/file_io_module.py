"""
File I/O Module - Read ECG (.dat) and PCG (.wav) files
"""

import numpy as np
import struct
import wave


def read_dat_as_ecg(dat_path, fs=2000, gain=200.0, baseline=0, num_channels=1):
    """
    Read ECG data from .dat file
    
    Parameters:
    -----------
    dat_path : str
        Path to .dat file
    fs : float
        Sampling frequency
    gain : float
        Signal gain
    baseline : float
        Baseline offset
    num_channels : int
        Number of channels
        
    Returns:
    --------
    physical_signals : ndarray
        Physical signal values
    fs : float
        Sampling frequency
    """
    with open(dat_path, 'rb') as f:
        data = f.read()
    num_values = len(data) // 2
    samples = struct.unpack(f'<{num_values}h', data)
    samples = np.array(samples)
    if num_channels > 1:
        num_samples = len(samples) // num_channels
        signals = samples[:num_samples * num_channels].reshape(num_samples, num_channels)
    else:
        signals = samples.reshape(-1, 1)
    physical_signals = (signals - baseline) / gain
    print(f"DAT: {physical_signals.shape[0] / fs:.2f}s")
    return physical_signals, fs


def read_pcg_wav(wav_path):
    """
    Read PCG data from .wav file
    
    Parameters:
    -----------
    wav_path : str
        Path to .wav file
        
    Returns:
    --------
    fs : int
        Sampling frequency
    data : ndarray
        Normalized audio data
    """
    with wave.open(wav_path, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        fs = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        frames = wav_file.readframes(n_frames)
        if sampwidth == 1:
            fmt = f'{n_frames * n_channels}B'
            normalizer = 128.0
            offset = -128
        elif sampwidth == 2:
            fmt = f'<{n_frames * n_channels}h'
            normalizer = 32768.0
            offset = 0
        elif sampwidth == 4:
            fmt = f'<{n_frames * n_channels}i'
            normalizer = 2147483648.0
            offset = 0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
        data = struct.unpack(fmt, frames)
        data = np.array(data, dtype=float)
        data = (data + offset) / normalizer
        if n_channels > 1:
            data = data.reshape(-1, n_channels)
            data = data[:, 0]
        print(f"WAV: {fs}Hz, {n_frames/fs:.2f}s")
        return fs, data