"""
Main GUI Application - ECG & PCG Analysis
Integrates all modules with interactive interface
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os

# Import custom modules
from file_io_module import read_dat_as_ecg, read_pcg_wav
from pantompkins_module import detect_r_peaks_with_fallback
from segmentation_module import segment_one_cycle
from cwt_module import compute_cwt_pascal
from stft_module import compute_stft
from cog_module import threshold_mask, compute_cog


class ECGPCGAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG & PCG Analysis System")
        self.root.geometry("1400x900")
        
        # Data storage
        self.ecg_signal = None
        self.ecg_fs = None
        self.pcg_signal = None
        self.pcg_fs = None
        self.r_peaks = None
        self.current_segment_idx = 0
        self.current_pcg_segment = None
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        """Create GUI widgets"""
        # ===== CONTROL PANEL =====
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File loading section
        file_frame = ttk.LabelFrame(control_frame, text="1. Load Files", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(file_frame, text="Load ECG (.dat)", 
                  command=self.load_ecg).grid(row=0, column=0, padx=5)
        self.ecg_label = ttk.Label(file_frame, text="No file loaded", foreground="gray")
        self.ecg_label.grid(row=0, column=1, padx=5)
        
        ttk.Button(file_frame, text="Load PCG (.wav)", 
                  command=self.load_pcg).grid(row=1, column=0, padx=5)
        self.pcg_label = ttk.Label(file_frame, text="No file loaded", foreground="gray")
        self.pcg_label.grid(row=1, column=1, padx=5)
        
        # Processing section
        process_frame = ttk.LabelFrame(control_frame, text="2. Process", padding="10")
        process_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(process_frame, text="Detect R-Peaks", 
                  command=self.detect_r_peaks, width=20).grid(row=0, column=0, padx=5, pady=5)
        self.rpeak_label = ttk.Label(process_frame, text="No R-peaks detected")
        self.rpeak_label.grid(row=0, column=1, padx=5)
        
        ttk.Button(process_frame, text="Show Overview", 
                  command=self.show_overview, width=20).grid(row=1, column=0, padx=5, pady=5)
        
        # Segmentation section
        seg_frame = ttk.LabelFrame(control_frame, text="3. Segmentation", padding="10")
        seg_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(seg_frame, text="Segment:").grid(row=0, column=0, padx=5)
        self.segment_var = tk.IntVar(value=0)
        self.segment_spinbox = ttk.Spinbox(seg_frame, from_=0, to=0, 
                                          textvariable=self.segment_var,
                                          width=10, command=self.update_segment)
        self.segment_spinbox.grid(row=0, column=1, padx=5)
        
        ttk.Button(seg_frame, text="◀ Prev", 
                  command=self.prev_segment, width=10).grid(row=0, column=2, padx=2)
        ttk.Button(seg_frame, text="Next ▶", 
                  command=self.next_segment, width=10).grid(row=0, column=3, padx=2)
        
        self.seg_info_label = ttk.Label(seg_frame, text="RR: -- ms | HR: -- BPM")
        self.seg_info_label.grid(row=1, column=0, columnspan=4, pady=5)
        
        # Analysis section
        analysis_frame = ttk.LabelFrame(control_frame, text="4. Analysis", padding="10")
        analysis_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(analysis_frame, text="CWT Analysis", 
                  command=self.show_cwt, width=20).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(analysis_frame, text="STFT Analysis", 
                  command=self.show_stft, width=20).grid(row=1, column=0, padx=5, pady=5)
        
        # CWT Parameters
        param_frame = ttk.LabelFrame(control_frame, text="CWT Parameters", padding="10")
        param_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(param_frame, text="S1 Threshold:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.s1_thr_var = tk.DoubleVar(value=0.6)
        ttk.Scale(param_frame, from_=0.3, to=0.9, variable=self.s1_thr_var,
                 orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=5)
        self.s1_thr_label = ttk.Label(param_frame, text="0.60")
        self.s1_thr_label.grid(row=0, column=2, padx=5)
        self.s1_thr_var.trace('w', self.update_s1_label)
        
        ttk.Label(param_frame, text="S2 Threshold:").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.s2_thr_var = tk.DoubleVar(value=0.4)
        ttk.Scale(param_frame, from_=0.2, to=0.8, variable=self.s2_thr_var,
                 orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, padx=5)
        self.s2_thr_label = ttk.Label(param_frame, text="0.40")
        self.s2_thr_label.grid(row=1, column=2, padx=5)
        self.s2_thr_var.trace('w', self.update_s2_label)
        
        ttk.Button(param_frame, text="Apply & Update", 
                  command=self.apply_thresholds, width=20).grid(row=2, column=0, 
                                                                columnspan=3, pady=10)
        
        # Status
        self.status_label = ttk.Label(control_frame, text="Ready", 
                                     relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # ===== PLOT AREA =====
        plot_frame = ttk.Frame(self.root)
        plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        
        # Configure grid weights
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def update_s1_label(self, *args):
        """Update S1 threshold label"""
        self.s1_thr_label.config(text=f"{self.s1_thr_var.get():.2f}")
        
    def update_s2_label(self, *args):
        """Update S2 threshold label"""
        self.s2_thr_label.config(text=f"{self.s2_thr_var.get():.2f}")
        
    def load_ecg(self):
        """Load ECG file"""
        filename = filedialog.askopenfilename(
            title="Select ECG file",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.ecg_signal, self.ecg_fs = read_dat_as_ecg(filename)
                self.ecg_signal = self.ecg_signal[:, 0]  # First channel
                self.ecg_label.config(text=os.path.basename(filename), foreground="green")
                self.status_label.config(text=f"ECG loaded: {len(self.ecg_signal)/self.ecg_fs:.2f}s")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load ECG: {str(e)}")
                
    def load_pcg(self):
        """Load PCG file"""
        filename = filedialog.askopenfilename(
            title="Select PCG file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.pcg_fs, self.pcg_signal = read_pcg_wav(filename)
                self.pcg_label.config(text=os.path.basename(filename), foreground="green")
                self.status_label.config(text=f"PCG loaded: {len(self.pcg_signal)/self.pcg_fs:.2f}s")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load PCG: {str(e)}")
                
    def detect_r_peaks(self):
        """Detect R-peaks in ECG"""
        if self.ecg_signal is None:
            messagebox.showwarning("Warning", "Please load ECG file first")
            return
            
        try:
            self.status_label.config(text="Detecting R-peaks...")
            self.root.update()
            
            self.r_peaks = detect_r_peaks_with_fallback(self.ecg_signal, fs=self.ecg_fs)
            
            if len(self.r_peaks) > 1:
                avg_rr = np.mean(np.diff(self.r_peaks)) / self.ecg_fs
                hr = 60 / avg_rr
                self.rpeak_label.config(text=f"{len(self.r_peaks)} peaks | HR: {hr:.1f} BPM")
                self.segment_spinbox.config(to=len(self.r_peaks)-2)
                self.status_label.config(text=f"R-peaks detected: {len(self.r_peaks)}")
            else:
                messagebox.showwarning("Warning", "Not enough R-peaks detected")
        except Exception as e:
            messagebox.showerror("Error", f"R-peak detection failed: {str(e)}")
            
    def show_overview(self):
        """Show overview plot"""
        if self.ecg_signal is None or self.pcg_signal is None:
            messagebox.showwarning("Warning", "Please load both ECG and PCG files")
            return
            
        if self.r_peaks is None or len(self.r_peaks) < 2:
            messagebox.showwarning("Warning", "Please detect R-peaks first")
            return
            
        self.figure.clear()
        
        duration = 10
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212, sharex=ax1)
        
        # ECG plot
        end_ecg = min(len(self.ecg_signal), int(duration * self.ecg_fs))
        ecg_time = np.arange(end_ecg) / self.ecg_fs
        ax1.plot(ecg_time, self.ecg_signal[:end_ecg], 'b', linewidth=0.8, label='ECG')
        
        r_in_view = self.r_peaks[self.r_peaks < end_ecg]
        ax1.scatter(r_in_view / self.ecg_fs, self.ecg_signal[r_in_view], 
                   c='r', s=40, zorder=3, marker='o', label='R-peaks')
        ax1.set_ylabel('ECG (mV)', fontsize=11)
        ax1.set_title('ECG Overview', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # PCG plot
        end_pcg = min(len(self.pcg_signal), int(duration * self.pcg_fs))
        pcg_time = np.arange(end_pcg) / self.pcg_fs
        ax2.plot(pcg_time, self.pcg_signal[:end_pcg], 'r', linewidth=0.8, label='PCG')
        
        for rp in r_in_view:
            ax2.axvline(rp / self.ecg_fs, color='orange', linestyle='--', 
                       alpha=0.5, linewidth=0.8)
        ax2.set_ylabel('PCG', fontsize=11)
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_title('PCG Overview', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        self.figure.tight_layout()
        self.canvas.draw()
        self.status_label.config(text="Overview displayed")
        
    def update_segment(self):
        """Update current segment"""
        if self.r_peaks is None or len(self.r_peaks) < 2:
            return
            
        self.current_segment_idx = self.segment_var.get()
        self.plot_segment()
        
    def prev_segment(self):
        """Go to previous segment"""
        if self.current_segment_idx > 0:
            self.current_segment_idx -= 1
            self.segment_var.set(self.current_segment_idx)
            self.plot_segment()
            
    def next_segment(self):
        """Go to next segment"""
        max_idx = len(self.r_peaks) - 2
        if self.current_segment_idx < max_idx:
            self.current_segment_idx += 1
            self.segment_var.set(self.current_segment_idx)
            self.plot_segment()
            
    def plot_segment(self):
        """Plot current segment"""
        if self.pcg_signal is None or self.r_peaks is None:
            return
            
        try:
            idx = self.current_segment_idx
            pcg_seg, start_pcg, end_pcg = segment_one_cycle(
                self.pcg_signal, self.r_peaks, idx, pad_ms=50.0, fs=self.pcg_fs)
            
            self.current_pcg_segment = pcg_seg
            
            start_ecg = int(start_pcg * self.ecg_fs / self.pcg_fs)
            end_ecg = int(end_pcg * self.ecg_fs / self.pcg_fs)
            ecg_seg = self.ecg_signal[start_ecg:end_ecg]
            
            self.figure.clear()
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212, sharex=ax1)
            
            # ECG
            ecg_time = np.arange(len(ecg_seg)) / self.ecg_fs
            ax1.plot(ecg_time, ecg_seg, 'b', linewidth=1.5, alpha=0.8)
            ax1.set_ylabel('ECG (mV)', fontsize=12, fontweight='bold', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # R-peaks in segment
            r_in_seg = self.r_peaks[(self.r_peaks >= start_ecg) & (self.r_peaks < end_ecg)]
            for rp in r_in_seg:
                rp_time = (rp - start_ecg) / self.ecg_fs
                ax1.axvline(rp_time, color='orange', linestyle='--', alpha=0.7, linewidth=2)
                ax1.scatter([rp_time], [ecg_seg[int((rp - start_ecg))]], 
                           c='red', s=80, zorder=5, marker='o')
            ax1.grid(True, alpha=0.3)
            ax1.set_title('ECG Segment', fontsize=11, fontweight='bold')
            
            # PCG
            pcg_time = np.arange(len(pcg_seg)) / self.pcg_fs
            ax2.plot(pcg_time, pcg_seg, 'r', linewidth=1.5, alpha=0.9)
            ax2.set_ylabel('PCG', fontsize=12, fontweight='bold', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_xlabel('Time (s)', fontsize=11)
            
            for rp in r_in_seg:
                rp_time = (rp - start_ecg) / self.ecg_fs
                ax2.axvline(rp_time, color='orange', linestyle='--', alpha=0.7, linewidth=2)
            ax2.grid(True, alpha=0.3)
            ax2.set_title('PCG Segment', fontsize=11, fontweight='bold')
            
            # Update info
            rr_interval = (self.r_peaks[idx+1] - self.r_peaks[idx]) / self.ecg_fs
            hr = 60.0 / rr_interval
            self.seg_info_label.config(text=f"RR: {rr_interval*1000:.1f} ms | HR: {hr:.1f} BPM")
            self.figure.suptitle(f'Segment {idx+1}/{len(self.r_peaks)-1}', 
                               fontsize=14, fontweight='bold')
            
            self.figure.tight_layout()
            self.canvas.draw()
            self.status_label.config(text=f"Displaying segment {idx+1}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot segment: {str(e)}")
            
    def show_cwt(self):
        """Show CWT analysis in new window"""
        if self.current_pcg_segment is None:
            messagebox.showwarning("Warning", "Please select a segment first")
            return
            
        self.show_analysis_window('CWT')
        
    def show_stft(self):
        """Show STFT analysis in new window"""
        if self.current_pcg_segment is None:
            messagebox.showwarning("Warning", "Please select a segment first")
            return
            
        self.show_analysis_window('STFT')
        
    def show_analysis_window(self, analysis_type):
        """Show analysis in new window"""
        window = tk.Toplevel(self.root)
        window.title(f"{analysis_type} Analysis - Segment {self.current_segment_idx+1}")
        window.geometry("1200x800")
        
        fig = Figure(figsize=(12, 8))
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()
        
        try:
            if analysis_type == 'CWT':
                scalogram, freqs, times = compute_cwt_pascal(
                    self.current_pcg_segment, self.pcg_fs, 
                    fmin=20.0, fmax=500.0, n_freqs=120)
                data = scalogram
                
                # Plot CWT with CoG
                ax = fig.add_subplot(111)
                extent = [times[0], times[-1], freqs[0], freqs[-1]]
                im = ax.imshow(data, aspect='auto', origin='lower', 
                              extent=extent, cmap='jet', interpolation='bilinear')
                
                # Compute and plot CoG
                thr_s1 = self.s1_thr_var.get()
                thr_s2 = self.s2_thr_var.get()
                
                mask_s1 = threshold_mask(data, thr_s1, min_area=5, keep_top=3)
                cog_s1 = compute_cog(data, freqs, times, mask=mask_s1)
                
                if mask_s1.any():
                    ax.contour(mask_s1.astype(float), levels=[0.5], colors='black', 
                              linewidths=3.5, alpha=0.9, extent=extent, origin='lower')
                
                if cog_s1 is not None:
                    t_s1, f_s1 = cog_s1
                    ax.plot(t_s1, f_s1, 'x', color='black', markersize=10, 
                           markeredgewidth=4, label=f'S1: {t_s1*1000:.1f}ms, {f_s1:.1f}Hz', zorder=10)
                
                mask_s2 = threshold_mask(data, thr_s2, min_area=5, keep_top=3)
                cog_s2 = compute_cog(data, freqs, times, mask=mask_s2)
                
                if mask_s2.any():
                    ax.contour(mask_s2.astype(float), levels=[0.5], colors='black', 
                              linewidths=3.5, alpha=0.9, extent=extent, origin='lower')
                
                if cog_s2 is not None:
                    t_s2, f_s2 = cog_s2
                    ax.plot(t_s2, f_s2, 'o', color='black', markersize=10, 
                           markeredgewidth=4, markerfacecolor='none',
                           label=f'S2: {t_s2*1000:.1f}ms, {f_s2:.1f}Hz', zorder=10)
                
                ax.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
                ax.set_xlabel('Time (s)', fontsize=11)
                ax.set_title(f'CWT - Segment {self.current_segment_idx+1}', 
                           fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.2, color='white', linestyle=':')
                ax.legend(loc='upper right', fontsize=10)
                
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Power', fontsize=11)
                
            else:  # STFT
                Sxx, freqs, times, _ = compute_stft(
                    self.current_pcg_segment, fs=self.pcg_fs, 
                    nperseg=256, noverlap=128, window='hann')
                
                maxv = np.nanmax(Sxx)
                if maxv > 0:
                    Sxx = Sxx / maxv
                data_plot = 20 * np.log10(Sxx + 1e-10)
                
                ax = fig.add_subplot(111)
                extent = [times[0], times[-1], freqs[0], freqs[-1]]
                im = ax.imshow(data_plot, aspect='auto', origin='lower', 
                              extent=extent, cmap='turbo', interpolation='nearest')
                
                ax.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
                ax.set_xlabel('Time (s)', fontsize=11)
                ax.set_title(f'STFT - Segment {self.current_segment_idx+1}', 
                           fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.2, color='white', linestyle=':')
                
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('dB', fontsize=11)
            
            fig.tight_layout()
            canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            
    def apply_thresholds(self):
        """Apply threshold changes and update CWT plot"""
        if self.current_pcg_segment is None:
            messagebox.showwarning("Warning", "Please select a segment first")
            return
        
        # Close any existing analysis windows and show new one
        self.show_cwt()
        self.status_label.config(text="Thresholds applied")


def main():
    root = tk.Tk()
    app = ECGPCGAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()