"""
Pan-Tompkins Module - R-peak Detection
Implementation of Pan-Tompkins QRS detection algorithm
"""

import numpy as np


def _design_bandpass_fir(fs, low, high, kernel_len=None):
    """Design bandpass FIR filter"""
    if kernel_len is None:
        kernel_len = int(round(min(1025, max(31, 0.128 * fs))))
    kernel_len = int(kernel_len)
    if kernel_len % 2 == 0:
        kernel_len += 1
    nyq = 0.5 * fs
    f1, f2 = float(low) / nyq, float(high) / nyq
    m = (kernel_len - 1) // 2
    n = np.arange(-m, m+1, dtype=float)
    h = (np.sinc(2 * f2 * n) - np.sinc(2 * f1 * n))
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * (np.arange(kernel_len) / float(kernel_len - 1))))
    h *= w
    denom = np.sum(np.abs(h))
    if denom <= 0: denom = 1.0
    h /= denom
    return h


def _filtfilt_like_linear_phase(sig, kernel):
    """Apply linear-phase filtering"""
    sig = np.asarray(sig, dtype=float)
    if sig.size == 0 or kernel.size == 0:
        return sig.copy()
    pad = len(kernel)
    if sig.size >= pad:
        left, right = sig[:pad][::-1], sig[-pad:][::-1]
    else:
        left, right = sig[::-1], sig[::-1]
    xpad = np.concatenate([left, sig, right])
    y = np.convolve(xpad, kernel, mode='same')
    return y[pad:pad + len(sig)]


def _five_point_derivative(sig, fs):
    """Five-point derivative"""
    kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0])
    deriv = np.convolve(sig, kernel, mode='same')
    deriv *= (fs / 8.0)
    return deriv


def _moving_window_integration(x, fs, window_ms=150.0):
    """Moving window integration"""
    win = max(1, int(round(window_ms / 1000.0 * fs)))
    kernel = np.ones(win) / float(win)
    return np.convolve(x, kernel, mode='same')


def _refine_to_ecg_peak(ecg, center_idx, fs, rad_ms=30.0):
    """Refine peak position to maximum in ECG"""
    rad = max(1, int(round(rad_ms / 1000.0 * fs)))
    lo, hi = max(0, center_idx - rad), min(len(ecg) - 1, center_idx + rad)
    segment = ecg[lo:hi+1]
    if segment.size == 0: return center_idx
    return lo + int(np.argmax(np.abs(segment)))


def _max_slope_around(ecg, idx, fs, window_ms=40):
    """Compute maximum slope around a point"""
    rad = max(1, int(round(window_ms / 1000.0 * fs)))
    lo, hi = max(0, idx - rad), min(len(ecg) - 1, idx + rad)
    seg = ecg[lo:hi+1]
    if seg.size < 2: return 0.0
    return float(np.max(np.abs(np.diff(seg))))


def _find_peaks_numpy(x, height=None, distance=None):
    """Find peaks in signal"""
    x = np.asarray(x)
    N = x.size
    if N == 0: return np.array([], dtype=int)
    left = np.r_[x[0] - 1e-12, x[:-1]]
    right = np.r_[x[1:], x[-1] - 1e-12]
    peaks_mask = (x > left) & (x >= right)
    peaks_idx = np.nonzero(peaks_mask)[0]
    if height is not None:
        peaks_idx = peaks_idx[x[peaks_idx] >= height]
    if distance is None or distance <= 1 or peaks_idx.size <= 1:
        return peaks_idx.astype(int)
    peaks_vals = x[peaks_idx]
    order = np.argsort(-peaks_vals)
    keep = np.zeros_like(peaks_idx, dtype=bool)
    taken = np.zeros(N, dtype=bool)
    for i in order:
        idx = peaks_idx[i]
        if not taken[idx]:
            keep[i] = True
            lo, hi = max(0, idx - distance), min(N, idx + distance + 1)
            taken[lo:hi] = True
    kept = peaks_idx[keep]
    kept.sort()
    return kept.astype(int)


def _robust_normalize(x):
    """Robust normalization using median and MAD"""
    x = np.asarray(x).astype(float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-6:
        std = np.std(x) if np.std(x) > 1e-6 else 1.0
        return (x - med) / std
    return (x - med) / (1.4826 * mad)


def _fallback_qrs_detector(ecg, fs):
    """Fallback QRS detector"""
    low, high = 1.0, 40.0
    try:
        kernel = _design_bandpass_fir(fs, low, high)
        filtered = _filtfilt_like_linear_phase(ecg, kernel)
    except:
        filtered = ecg
    env = np.abs(filtered)
    med = np.median(env)
    thresh = med + 0.5 * np.std(env)
    distance = int(round(0.18 * fs))
    return _find_peaks_numpy(env, height=thresh, distance=distance)


def _amplitude_stabilize(ecg, fs, env_win_ms=200.0, max_ratio_thresh=8.0, gain_clip=(0.3, 3.0)):
    """Amplitude stabilization"""
    ecg = np.asarray(ecg, dtype=float)
    win = max(1, int(round(env_win_ms / 1000.0 * fs)))
    kernel = np.ones(win) / float(win)
    E = np.convolve(np.abs(ecg), kernel, mode='same')
    med = np.median(E) if E.size > 0 else 0.0
    eps = 1e-12
    if med <= 0: med = np.mean(E) + 1e-9
    ratio = np.max(E) / (med + eps) if E.size > 0 else 1.0
    if ratio <= max_ratio_thresh:
        return ecg, False
    gain = med / (E + eps)
    gain = np.clip(gain, gain_clip[0], gain_clip[1])
    return ecg * gain, True


def detect_r_peaks(ecg, fs=2000, low_hz=5.0, high_hz=15.0, integration_ms=150.0, 
                   refractory_ms=200.0, search_back_factor=1.66, debug=False):
    """
    Detect R-peaks using Pan-Tompkins algorithm
    
    Parameters:
    -----------
    ecg : array-like
        ECG signal
    fs : float
        Sampling frequency
    low_hz : float
        Low cutoff frequency for bandpass filter
    high_hz : float
        High cutoff frequency for bandpass filter
    integration_ms : float
        Moving window integration window (ms)
    refractory_ms : float
        Refractory period (ms)
    search_back_factor : float
        Search back factor for missed beats
    debug : bool
        Debug mode
        
    Returns:
    --------
    r_peaks : ndarray
        Array of R-peak indices
    """
    ecg = np.asarray(ecg).ravel()
    N = ecg.size
    if N == 0: return np.array([], dtype=int)
    try:
        ecg_proc, _ = _amplitude_stabilize(ecg, fs)
    except:
        ecg_proc = ecg.copy()
    try:
        kernel = _design_bandpass_fir(fs, low_hz, high_hz)
        ecg_f = _filtfilt_like_linear_phase(ecg_proc, kernel)
    except:
        ecg_f = ecg_proc.copy()
    deriv = _five_point_derivative(ecg_f, fs)
    squared = deriv ** 2
    mwi = _moving_window_integration(squared, fs, integration_ms)
    min_dist = max(1, int(round(refractory_ms / 1000.0 * fs)))
    cand_idxs = _find_peaks_numpy(mwi, distance=min_dist)
    if cand_idxs.size == 0: return np.array([], dtype=int)
    n_init = min(8, cand_idxs.size)
    init_idxs = cand_idxs[:n_init]
    init_i = mwi[init_idxs] if init_idxs.size > 0 else np.array([0.0])
    init_f = np.abs(ecg_f[init_idxs]) if init_idxs.size > 0 else np.array([0.0])
    SPKI = float(np.max(init_i)) if init_i.size > 0 else 1.0
    NPKI = float(np.median(init_i) * 0.5) if init_i.size > 0 else SPKI * 0.125
    SPKF = float(np.max(init_f)) if init_f.size > 0 else 1.0
    NPKF = float(np.median(init_f) * 0.5) if init_f.size > 0 else SPKF * 0.125
    THRESHOLD_I = NPKI + 0.25 * (SPKI - NPKI)
    THRESHOLD_F = NPKF + 0.25 * (SPKF - NPKF)
    THRESHOLD_I2, THRESHOLD_F2 = 0.5 * THRESHOLD_I, 0.5 * THRESHOLD_F
    detected = []
    for idx_cand in cand_idxs:
        val_i, val_f = float(mwi[idx_cand]), float(np.abs(ecg_f[idx_cand]))
        if detected and (idx_cand - detected[-1]) < min_dist:
            NPKI = 0.125 * val_i + 0.875 * NPKI
            NPKF = 0.125 * val_f + 0.875 * NPKF
            THRESHOLD_I = NPKI + 0.25 * (SPKI - NPKI)
            THRESHOLD_F = NPKF + 0.25 * (SPKF - NPKF)
            THRESHOLD_I2, THRESHOLD_F2 = 0.5 * THRESHOLD_I, 0.5 * THRESHOLD_F
            continue
        accept, threshold_used = False, None
        if (val_i >= THRESHOLD_I) or (val_f >= THRESHOLD_F):
            accept, threshold_used = True, 'primary'
        elif (val_i >= THRESHOLD_I2) or (val_f >= THRESHOLD_F2):
            accept, threshold_used = True, 'secondary'
        if accept:
            refined = _refine_to_ecg_peak(ecg, idx_cand, fs)
            is_t_wave = False
            if detected:
                prev = detected[-1]
                dt = (refined - prev) / float(fs)
                if dt < 0.36:
                    curr_slope = _max_slope_around(ecg, refined, fs)
                    prev_slope = _max_slope_around(ecg, prev, fs)
                    if prev_slope > 0 and curr_slope < 0.5 * prev_slope:
                        is_t_wave = True
            if is_t_wave:
                NPKI = 0.125 * val_i + 0.875 * NPKI
                NPKF = 0.125 * val_f + 0.875 * NPKF
            else:
                if not detected or (refined - detected[-1]) >= min_dist:
                    detected.append(int(refined))
                    if threshold_used == 'primary':
                        SPKI = 0.125 * val_i + 0.875 * SPKI
                        SPKF = 0.125 * val_f + 0.875 * SPKF
                    else:
                        SPKI = 0.25 * val_i + 0.75 * SPKI
                        SPKF = 0.25 * val_f + 0.75 * SPKF
                else:
                    NPKI = 0.125 * val_i + 0.875 * NPKI
                    NPKF = 0.125 * val_f + 0.875 * NPKF
        else:
            NPKI = 0.125 * val_i + 0.875 * NPKI
            NPKF = 0.125 * val_f + 0.875 * NPKF
        THRESHOLD_I = NPKI + 0.25 * (SPKI - NPKI)
        THRESHOLD_F = NPKF + 0.25 * (SPKF - NPKF)
        THRESHOLD_I2, THRESHOLD_F2 = 0.5 * THRESHOLD_I, 0.5 * THRESHOLD_F
    detected = np.array(detected, dtype=int)
    if detected.size >= 2:
        rr_ms = np.diff(detected)
        if rr_ms.size > 0:
            RRavg = int(np.mean(rr_ms[-8:]))
            if RRavg > 0:
                RRmiss = int(round(search_back_factor * RRavg))
                new_found = []
                for i in range(len(detected) - 1):
                    a, b = detected[i], detected[i+1]
                    gap = b - a
                    if gap > RRmiss:
                        lo, hi = a + 1, b - 1
                        if lo >= hi: continue
                        local_idx_rel = int(np.argmax(mwi[lo:hi+1]))
                        local_idx = lo + local_idx_rel
                        if mwi[local_idx] >= THRESHOLD_I2:
                            refined = _refine_to_ecg_peak(ecg, local_idx, fs)
                            new_found.append(int(refined))
                if new_found:
                    detected = np.unique(np.concatenate([detected, np.array(new_found, dtype=int)]))
                    detected.sort()
    if detected.size > 1:
        final = []
        i = 0
        while i < detected.size:
            block = [detected[i]]
            j = i + 1
            while j < detected.size and (detected[j] - detected[i]) < min_dist:
                block.append(detected[j])
                j += 1
            if len(block) == 1:
                final.append(block[0])
            else:
                vals = [abs(ecg[idx]) for idx in block]
                final.append(block[int(np.argmax(vals))])
            i = j
        detected = np.array(final, dtype=int)
    detected.sort()
    return detected


def detect_r_peaks_with_fallback(ecg, fs=2000, debug=False):
    """
    Detect R-peaks with fallback method
    
    Parameters:
    -----------
    ecg : array-like
        ECG signal
    fs : float
        Sampling frequency
    debug : bool
        Debug mode
        
    Returns:
    --------
    r_peaks : ndarray
        Array of R-peak indices
    """
    ecg_norm = _robust_normalize(ecg)
    try:
        r = detect_r_peaks(ecg_norm, fs=fs, debug=debug)
    except:
        r = np.array([], dtype=int)
    if r is None or len(r) < 2:
        fb = _fallback_qrs_detector(ecg_norm, fs)
        if len(fb) >= 2:
            refined = []
            for p in fb:
                lo, hi = max(0, p - int(0.02 * fs)), min(len(ecg_norm) - 1, p + int(0.02 * fs))
                sub = ecg_norm[lo:hi+1]
                if sub.size == 0: continue
                refined.append(lo + int(np.argmax(np.abs(sub))))
            r = np.unique(np.array(refined, dtype=int))
    if r is None:
        return np.array([], dtype=int)
    return np.array(sorted(np.unique(r)), dtype=int)