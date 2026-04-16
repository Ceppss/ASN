import numpy as np
import matplotlib.pyplot as plt
import load
import bpf
import dwt
import cwt
import stft
import tressholding


header_path = "D:/Semester 5/4. ASN/ASN Pak Adib/EMG/S01.hea"  
dat_path = "D:/Semester 5/4. ASN/ASN Pak Adib/EMG/S01.dat"      
fs = 2000

df, gait_cycle, ch10_cycle, ch13_cycle, segment_range = load.process_emg(header_path, dat_path)

if gait_cycle is None:
    print("Segmentasi dibatalkan. Program berhenti.")
    exit()

channels = { "ch10": ch10_cycle, "ch13": ch13_cycle}
filtered = bpf.apply_filter_to_channels(channels, 20, 450, fs)
bpf.plot_filtered_channels(channels, filtered, fs)

levels_gl = dwt.dwt_multilevel(filtered["ch10"], 8)
levels_vl = dwt.dwt_multilevel(filtered["ch13"], 8)

dwt.plot_dwt(levels_vl, levels_gl, fs)

selected_levels = ["D4"]

VL_band = dwt.idwt(levels_vl, selected_levels)
GL_band = dwt.idwt(levels_gl, selected_levels)

freqs = np.linspace(20, 450, 120)

coeffs_vl, freqs_out = cwt.cwt_morlet(VL_band, fs, freqs)
coeffs_gl, _ = cwt.cwt_morlet(GL_band, fs, freqs)

cwt.plot_scalograms_two_signals(coeffs_vl, coeffs_gl, freqs_out, fs)

#cwt.plot_cwt_3d(coeffs_vl, freqs_out, fs, title="3D CWT - VL")
#cwt.plot_cwt_3d(coeffs_gl, freqs_out, fs, title="3D CWT - GL")

win_size = 256
hop_size = 128

stft_vl, f_vl, t_vl = stft.stft(VL_band, fs, win_size, hop_size)
stft.plot_stft_spectrogram(stft_vl, f_vl, t_vl, title="STFT VL")

stft_gl, f_gl, t_gl = stft.stft(GL_band, fs, win_size, hop_size)
stft.plot_stft_spectrogram(stft_gl, f_gl, t_gl, title="STFT GL")


bin1, thr1, freq1 = tressholding.binary_threshold(coeffs_vl, freqs,thr=0.1)
bin2, thr2, freq2 = tressholding.binary_threshold(coeffs_gl, freqs,thr=0.1)

tressholding.plot_binary_threshold_dual(bin1, thr1, freq1,
                           bin2, thr2, freq2)

