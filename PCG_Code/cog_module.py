"""
CoG Detection Module - Center of Gravity Detection
Includes thresholding, masking, and component labeling
"""

import numpy as np


def _ensure_mask_shape(mask, target_shape):
    """Ensure mask has correct shape"""
    if mask is None:
        return np.zeros(target_shape, dtype=bool)
    mask = np.asarray(mask)
    if mask.shape == target_shape:
        return mask.astype(bool)
    if mask.T.shape == target_shape:
        return mask.T.astype(bool)
    return np.zeros(target_shape, dtype=bool)


def _label_components(bin_mask):
    """Label connected components using 8-connectivity flood-fill"""
    mask = np.asarray(bin_mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label = 0
    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    
    for i in range(h):
        for j in range(w):
            if not mask[i, j] or labels[i, j] != 0:
                continue
            label += 1
            stack = [(i, j)]
            labels[i, j] = label
            while stack:
                y, x = stack.pop()
                for dy, dx in neigh:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = label
                        stack.append((ny, nx))
    return labels, label


def threshold_mask(scalogram, thr_ratio, min_area=None, keep_top=3):
    """
    Create binary mask from scalogram based on threshold
    
    Parameters:
    -----------
    scalogram : ndarray
        2D scalogram array
    thr_ratio : float
        Threshold ratio (0-1)
    min_area : int, optional
        Minimum component area
    keep_top : int
        Number of top components to keep
        
    Returns:
    --------
    mask : ndarray
        Binary mask
    """
    S = np.asarray(scalogram)
    if S.ndim != 2:
        raise ValueError("scalogram must be 2D")
    
    maxS = np.nanmax(S)
    if maxS == 0 or np.isnan(maxS):
        return np.zeros_like(S, dtype=bool)
    
    global_thr = float(thr_ratio) * float(maxS)
    base_mask = (S >= global_thr)
    
    labeled, ncomp = _label_components(base_mask)
    if ncomp == 0:
        return np.zeros_like(S, dtype=bool)
    
    # Compute areas
    areas = []
    for idx in range(1, ncomp+1):
        ys, xs = np.nonzero(labeled == idx)
        areas.append(len(ys))
    areas = np.array(areas, dtype=int)
    
    if min_area is None:
        min_area_adaptive = max(3, int(0.005 * S.size))
    else:
        min_area_adaptive = int(min_area)
    
    # Keep components above min area
    keep_mask = np.zeros_like(S, dtype=bool)
    for idx, area in enumerate(areas, start=1):
        if area >= min_area_adaptive:
            keep_mask |= (labeled == idx)
    
    # If no components, keep top N largest
    if not keep_mask.any() and len(areas) > 0:
        order = np.argsort(-areas)
        for k in range(min(keep_top, len(order))):
            comp_idx = int(order[k]) + 1
            keep_mask |= (labeled == comp_idx)
    
    return keep_mask.astype(bool)


def compute_cog(scalogram, freqs, times, mask=None):
    """
    Compute Center of Gravity (CoG) from masked scalogram
    
    Parameters:
    -----------
    scalogram : ndarray
        2D scalogram array
    freqs : ndarray
        Frequency array
    times : ndarray
        Time array
    mask : ndarray, optional
        Binary mask
        
    Returns:
    --------
    cog : tuple or None
        (time_cog, freq_cog) or None if failed
    """
    S = np.asarray(scalogram, dtype=float)
    if S.ndim != 2:
        raise ValueError("scalogram must be 2D")
    
    n_freqs, n_times = S.shape
    freqs = np.asarray(freqs, dtype=float)
    times = np.asarray(times, dtype=float)
    
    if mask is None:
        M = np.ones_like(S, dtype=bool)
    else:
        M = _ensure_mask_shape(mask, (n_freqs, n_times))
    
    # Weighted sum
    W = S * M
    total = np.sum(W)
    
    if total <= 1e-12:
        return None
    
    # Create meshgrid for vectorized computation
    T, F = np.meshgrid(times, freqs)
    
    # Compute CoG
    t_cog = float(np.sum(W * T) / total)
    f_cog = float(np.sum(W * F) / total)
    
    return (t_cog, f_cog)