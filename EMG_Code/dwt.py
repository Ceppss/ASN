import numpy as np
import matplotlib.pyplot as plt


h = np.array([
    -0.0105974017850021,
     0.0328830116668852,
     0.0308413818355607,
    -0.1870348117188811,
    -0.0279837694169839,
     0.6308807678398578,
     0.7148465705529154,
     0.2303778133088552
])

g = np.array([
    -0.2303778133088552,
     0.7148465705529154,
    -0.6308807678398578,
    -0.0279837694169839,
     0.1870348117188811,
     0.0308413818355607,
    -0.0328830116668852,
    -0.0105974017850021
])

h_rev = h[::-1]
g_rev = g[::-1]

def dwt_single(x):

    A = np.convolve(x, h_rev, mode='same')[::2]
    D = np.convolve(x, g_rev, mode='same')[::2]
    return A, D


def upsample(x):
    u = np.zeros(len(x) * 2)
    u[::2] = x
    return u


def idwt_single(A, D):

    A_up = upsample(A)
    D_up = upsample(D)

    recA = np.convolve(A_up, h, mode='full')
    recD = np.convolve(D_up, g, mode='full')

    L = min(len(recA), len(recD))
    return recA[:L] + recD[:L]

def dwt_multilevel(x, level):

    results = {}
    A = x.copy()

    for L in range(1, level+1):
        A, D = dwt_single(A)
        results[f"A{L}"] = A
        results[f"D{L}"] = D

    return results

def idwt_multilevel(results, level):

    A = results[f"A{level}"]

    for L in range(level, 0, -1):
        D = results[f"D{L}"]
        A = idwt_single(A, D)

    return A


def plot_dwt(levels_vl, levels_gl, fs=2000):
    levels = len(levels_vl) // 2  
    rows = levels
    cols = 4  

    plt.figure(figsize=(18, 4 * levels))

    for L in range(1, levels + 1):
        A_vl = levels_vl[f"A{L}"]
        D_vl = levels_vl[f"D{L}"]
        
        A_gl = levels_gl[f"A{L}"]
        D_gl = levels_gl[f"D{L}"]

        tA_vl = np.arange(len(A_vl)) / fs
        tD_vl = np.arange(len(D_vl)) / fs
        tA_gl = np.arange(len(A_gl)) / fs
        tD_gl = np.arange(len(D_gl)) / fs

        plt.subplot(rows, cols, (L-1)*cols + 1)
        plt.plot(tA_vl, A_vl)
        plt.title(f"VL - A{L}")
        plt.grid(True)

        plt.subplot(rows, cols, (L-1)*cols + 2)
        plt.plot(tA_gl, A_gl)
        plt.title(f"GL - A{L}")
        plt.grid(True)

        plt.subplot(rows, cols, (L-1)*cols + 3)
        plt.plot(tD_vl, D_vl)
        plt.title(f"VL - D{L}")
        plt.grid(True)

        plt.subplot(rows, cols, (L-1)*cols + 4)
        plt.plot(tD_gl, D_gl)
        plt.title(f"GL - D{L}")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    
def idwt(results, selected_levels):
    level = max(int(k[1:]) for k in results if k.startswith("A"))
    A = np.zeros_like(results["A" + str(level)])  # hilangkan komponen A terakhir

    for L in range(level, 0, -1):
        A_L = A
        D_L = results[f"D{L}"] if f"D{L}" in selected_levels else np.zeros_like(results[f"D{L}"])
        A = idwt_single(A_L, D_L)

    return A
