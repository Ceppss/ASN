"""
Segmentation Module - PCG Signal Segmentation
"""

import numpy as np


def segment_one_cycle(pcg, r_peaks, idx=0, pad_ms=50.0, fs=2000, pre_ms=None, post_ms=None,
                      enforce_single_cycle=True, allow_partial_edges=True):
    """
    Segment one cardiac cycle from PCG signal
    
    Parameters:
    -----------
    pcg : array-like
        PCG signal
    r_peaks : array-like
        R-peak indices
    idx : int
        Cycle index
    pad_ms : float
        Padding in milliseconds
    fs : float
        Sampling frequency
    pre_ms : float, optional
        Pre-padding (overrides pad_ms)
    post_ms : float, optional
        Post-padding (overrides pad_ms)
    enforce_single_cycle : bool
        Enforce single cycle boundary
    allow_partial_edges : bool
        Allow partial edges
        
    Returns:
    --------
    segment : ndarray
        Segmented PCG
    start : int
        Start index
    end : int
        End index
    """
    if idx < 0 or idx >= len(r_peaks) - 1:
        raise IndexError("idx must be within 0..len(r_peaks)-2")
    if pre_ms is None: pre_ms = pad_ms
    if post_ms is None: post_ms = pad_ms
    pre_samp = int(round(pre_ms / 1000.0 * fs))
    post_samp = int(round(post_ms / 1000.0 * fs))
    start, end = r_peaks[idx] - pre_samp, r_peaks[idx + 1] + post_samp
    start, end = int(max(0, start)), int(min(len(pcg), end))
    if enforce_single_cycle:
        prev_mid = int(round(0.5 * (r_peaks[idx - 1] + r_peaks[idx]))) if idx > 0 else 0
        next_mid = int(round(0.5 * (r_peaks[idx + 1] + r_peaks[idx + 2]))) if idx + 2 < len(r_peaks) else len(pcg)
        start, end = max(start, prev_mid), min(end, next_mid)
        if start >= end:
            start, end = r_peaks[idx], r_peaks[idx + 1]
            start, end = max(0, start), min(len(pcg), end)
    start = int(max(0, min(start, len(pcg))))
    end = int(max(0, min(end, len(pcg))))
    return pcg[start:end].copy(), start, end