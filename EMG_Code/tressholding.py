import numpy as np
import matplotlib.pyplot as plt

def binary_threshold(coeffs, freqs,thr):
    mag = np.abs(coeffs)


    if mag.ndim == 1:
        mag_1d = mag
        auto_freq = freqs[0] if len(freqs) > 0 else 0

    else:
        energy = np.sum(mag, axis=1)
        idx = np.argmax(energy)
        auto_freq = freqs[idx]
        mag_1d = mag[idx, :]   

    thr = thr * np.max(mag_1d)

    binary_signal = np.where(mag_1d >= thr, 1, 0)

    return binary_signal, thr, auto_freq

def binary_threshold_dual(coeffs_1, coeffs_2, freqs):


    bin1, thr1, f1 = binary_threshold(coeffs_1, freqs)
    bin2, thr2, f2 = binary_threshold(coeffs_2, freqs)

    return bin1, bin2, thr1, thr2, f1, f2

def plot_binary_threshold_dual(bin_sig1, thr1, freq1,
                               bin_sig2, thr2, freq2):


    t = np.linspace(0, 100, len(bin_sig1))


    for i in range(1, len(bin_sig1)):
        if bin_sig1[i-1] == 0 and bin_sig1[i] == 1:
            print(f"Onset VL at {t[i]:.2f}%")
        if bin_sig1[i-1] == 1 and bin_sig1[i] == 0:
            print(f"Offset VL at {t[i]:.2f}%")

    for i in range(1, len(bin_sig2)):
        if bin_sig2[i-1] == 0 and bin_sig2[i] == 1:
            print(f"Onset GL at {t[i]:.2f}%")
        if bin_sig2[i-1] == 1 and bin_sig2[i] == 0:
            print(f"Offset GL at {t[i]:.2f}%")

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axs[0].plot(t, bin_sig1, linewidth=2)
    axs[0].fill_between(t, bin_sig1, alpha=0.3)
    axs[0].set_title("Binary Activation VL")
    axs[0].set_ylabel("State (0/1)")
    axs[0].set_ylim(-0.2, 1.2)
    axs[0].grid(True, linestyle="--", alpha=0.4)

    axs[1].plot(t, bin_sig2, linewidth=2)
    axs[1].fill_between(t, bin_sig2, alpha=0.3)
    axs[1].set_title("Binary Activation GL")
    axs[1].set_ylabel("State (0/1)")
    axs[1].set_xlabel("% Gait Cycle")
    axs[1].set_ylim(-0.2, 1.2)
    axs[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()






