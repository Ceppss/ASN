import numpy as np
import matplotlib.pyplot as plt

def design_bandpass_biquad(lowcut, highcut, fs):
    f0 = np.sqrt(lowcut * highcut)
    bw = highcut - lowcut
    Q = f0 / bw

    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)

    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a

def lfilter_direct(b, a, x):
    x = np.asarray(x, dtype=float)
    N = x.size
    y = np.zeros(N)

    x_1 = x_2 = 0.0
    y_1 = y_2 = 0.0
    b0, b1, b2 = b
    a1, a2 = a[1], a[2]

    for n in range(N):
        xn = x[n]
        yn = b0*xn + b1*x_1 + b2*x_2 - a1*y_1 - a2*y_2
        y[n] = yn
        x_2 = x_1
        x_1 = xn
        y_2 = y_1
        y_1 = yn
    return y

def filtfilt_manual(b, a, x):
    x = np.asarray(x, dtype=float)
    N = x.size
    padlen = 6  

    if N <= padlen:
        y = lfilter_direct(b, a, x)
        return lfilter_direct(b, a, y[::-1])[::-1]

    pre = 2*x[0] - x[1:padlen+1][::-1]
    post = 2*x[-1] - x[-padlen-1:-1][::-1]
    xs = np.concatenate([pre, x, post])

    yf = lfilter_direct(b, a, xs)
    yfb = lfilter_direct(b, a, yf[::-1])[::-1]
    return yfb[padlen:padlen+N]

def apply_filter_to_channels(channels_dict, lowcut=20, highcut=450, fs=2000):
    b, a = design_bandpass_biquad(lowcut, highcut, fs)
    out = {}
    for name, sig in channels_dict.items():
        out[name] = filtfilt_manual(b, a, sig)
    return out

def plot_filtered_channels(raw_dict, filtered_dict, fs=2000):
    fig, axs = plt.subplots(len(raw_dict), 1, figsize=(12, 8), sharex=True)

    if len(raw_dict) == 1:
        axs = [axs]

    for i, (name, raw_sig) in enumerate(raw_dict.items()):
        axs[i].plot(raw_sig, label=f"{name} - raw", alpha=0.6)
        axs[i].plot(filtered_dict[name], label=f"{name} - filtered", linewidth=2)
        axs[i].set_title(f"{name} (fs={fs} Hz)")
        axs[i].legend()

    plt.tight_layout()
    plt.show()
