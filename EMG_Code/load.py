import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt

def read_header(header_path):
    with open(header_path) as f:
        lines = f.read().strip().splitlines()

    gains = []
    baselines = []

    for line in lines[1:]:
        parts = line.split()
        m = re.match(r'([0-9.]+)\((-?\d+)\)/', parts[2])
        gains.append(float(m.group(1)))
        baselines.append(int(m.group(2)))

    return np.array(gains), np.array(baselines)

def read_raw_data(dat_path, gains, baselines, n_channels=14):
    raw = np.fromfile(dat_path, dtype=np.int16).reshape(-1, n_channels)
    p = (raw - baselines) / gains
    return raw, p

def extract_first_samples(p):
    df = pd.DataFrame({
        "channel_7": p[:20, 7],
        "channel_10": p[:20, 10],
        "channel_13": p[:20, 13]
    })
    return df

class ZoomSelector:

    
    def __init__(self, ch7, ch10, ch13):
        self.ch7 = ch7
        self.ch10 = ch10
        self.ch13 = ch13
        self.points = []
        self.fig = None
        self.axs = None
        self.scatter_points = []
        self.vlines = []
        self.span = None
        
    def on_click(self, event):
        if event.inaxes not in self.axs:
            return
            
        # Klik kanan untuk reset
        if event.button == 3:
            self.points = []
            for sp in self.scatter_points:
                sp.remove()
            for vl in self.vlines:
                vl.remove()
            if self.span:
                self.span.remove()
                self.span = None
            self.scatter_points = []
            self.vlines = []
            self.axs[0].set_title("Channel 7 (GAIT)")
            self.fig.canvas.draw()
            return
                        
        if event.button == 1:
            if len(self.points) >= 2:
                print("Sudah ada 2 titik. Klik kanan untuk reset.")
                return
                
            x_click = event.xdata
            idx = int(round(x_click))
            idx = max(0, min(idx, len(self.ch7) - 1))
            
            self.points.append(idx)
            
            # Tambah marker di semua 3 channel
            channels = [self.ch7, self.ch10, self.ch13]
            colors = ['blue', 'green', 'red']
            for ax, ch, c in zip(self.axs, channels, colors):
                sp = ax.scatter(idx, ch[idx], color='orange', s=150, 
                               zorder=5, marker='o', edgecolors='black', linewidths=2)
                vl = ax.axvline(x=idx, color='orange', linestyle='--', alpha=0.7)
                self.scatter_points.append(sp)
                self.vlines.append(vl)
            
            if len(self.points) == 1:
                self.axs[0].set_title(f"Channel 7 (GAIT)")
            else:
                self.points.sort()
                # Tambah highlight area yang dipilih
                for ax in self.axs:
                    self.span = ax.axvspan(self.points[0], self.points[1], 
                                          alpha=0.2, color='yellow')
                self.axs[0].set_title(f"Channel 7 (GAIT)")

                
            self.fig.canvas.draw()
    
    def select_zoom_area(self):
        self.fig, self.axs = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        
        self.axs[0].plot(self.ch7, 'b-', linewidth=0.5)
        self.axs[0].set_ylabel('Amplitude')
        self.axs[0].set_title("Channel 7 (GAIT)")
        self.axs[0].grid(True, alpha=0.3)
        
        self.axs[1].plot(self.ch10, 'g-', linewidth=0.5)
        self.axs[1].set_ylabel('Amplitude')
        self.axs[1].set_title("Channel 10 (VL)")
        self.axs[1].grid(True, alpha=0.3)
        
        self.axs[2].plot(self.ch13, 'r-', linewidth=0.5)
        self.axs[2].set_ylabel('Amplitude')
        self.axs[2].set_xlabel('Sample Index')
        self.axs[2].set_title("Channel 13 (GL)")
        self.axs[2].grid(True, alpha=0.3)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.tight_layout()
        plt.show()
        
        if len(self.points) == 2:
            return self.points[0], self.points[1]
        else:
            return None, None

class SegmentSelector:
    
    def __init__(self, ch7, ch10, ch13, zoom_start, zoom_end):
        self.ch7_zoomed = ch7[zoom_start:zoom_end]
        self.ch10_zoomed = ch10[zoom_start:zoom_end]
        self.ch13_zoomed = ch13[zoom_start:zoom_end]
        self.zoom_start = zoom_start
        self.zoom_end = zoom_end
        self.points = []
        self.fig = None
        self.axs = None
        self.scatter_points = []
        self.vlines = []
        
    def on_click(self, event):
        if event.inaxes not in self.axs:
            return
        if event.button == 3:
            self.points = []
            for sp in self.scatter_points:
                sp.remove()
            for vl in self.vlines:
                vl.remove()
            self.scatter_points = []
            self.vlines = []
            self.axs[0].set_title(f"Channel 7 (GAIT) - ZOOMED ")
            self.fig.canvas.draw()
            return
        if event.button == 1:
            
            x_click = event.xdata
            idx_rel = int(round(x_click))
            idx_rel = max(0, min(idx_rel, len(self.ch7_zoomed) - 1))
            self.points.append(idx_rel)
            channels = [self.ch7_zoomed, self.ch10_zoomed, self.ch13_zoomed]
            for ax, ch in zip(self.axs, channels):
                sp = ax.scatter(idx_rel, ch[idx_rel], color='red', s=200, 
                               zorder=5, marker='o', edgecolors='black', linewidths=2)
                vl = ax.axvline(x=idx_rel, color='red', linestyle='--', linewidth=2, alpha=0.8)
                self.scatter_points.append(sp)
                self.vlines.append(vl)

            idx_abs = self.zoom_start + idx_rel
            
            if len(self.points) == 1:
                self.axs[0].set_title(f"Channel 7 (GAIT) - ZOOMED")
            else:
                self.points.sort()
                abs_start = self.zoom_start + self.points[0]
                abs_end = self.zoom_start + self.points[1]
                
                for ax in self.axs:
                    ax.axvspan(self.points[0], self.points[1], alpha=0.3, color='lightgreen')
                
                self.axs[0].set_title(f"Channel 7 (GAIT) - ZOOMED")

            self.fig.canvas.draw()
    
    def select_segment(self):
        self.fig, self.axs = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        
        x_axis = np.arange(len(self.ch7_zoomed))
        
        self.axs[0].plot(x_axis, self.ch7_zoomed, 'b-', linewidth=1)
        self.axs[0].set_ylabel('Amplitude')
        self.axs[0].set_title(f"Channel 7 (GAIT) - ZOOMED ")
        self.axs[0].grid(True, alpha=0.3)
        
        self.axs[1].plot(x_axis, self.ch10_zoomed, 'g-', linewidth=1)
        self.axs[1].set_ylabel('Amplitude')
        self.axs[1].set_title("Channel 10 (VL) - ZOOMED")
        self.axs[1].grid(True, alpha=0.3)
        
        self.axs[2].plot(x_axis, self.ch13_zoomed, 'r-', linewidth=1)
        self.axs[2].set_ylabel('Amplitude')
        self.axs[2].set_xlabel(f'Sample Index (relatif terhadap zoom, +{self.zoom_start} untuk absolut)')
        self.axs[2].set_title("Channel 13 (GL) - ZOOMED")
        self.axs[2].grid(True, alpha=0.3)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.tight_layout()
        plt.show()
        
        if len(self.points) == 2:
            # Return indeks absolut
            abs_start = self.zoom_start + self.points[0]
            abs_end = self.zoom_start + self.points[1]
            return abs_start, abs_end
        else:
            return None, None


def zoom_and_segment(ch7, ch10, ch13):
    
    zoom_selector = ZoomSelector(ch7, ch10, ch13)
    zoom_start, zoom_end = zoom_selector.select_zoom_area()
    
    if zoom_start is None:
        print("Zoom area tidak dipilih. Proses dibatalkan.")
        return None, None
    
    print(f"\nZoom area: {zoom_start} - {zoom_end}")
    
    segment_selector = SegmentSelector(ch7, ch10, ch13, zoom_start, zoom_end)
    start_idx, end_idx = segment_selector.select_segment()
    
    return start_idx, end_idx

def plot_segmented_gait_cycle(gait_cycle, ch10_cycle, ch13_cycle, start_idx, end_idx):
    n_samples = len(gait_cycle)
    time_axis = np.arange(n_samples)
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    axs[0].plot(time_axis, gait_cycle, 'b-', linewidth=1)
    axs[0].axvline(x=0, color='green', linestyle='--', linewidth=2, label='Start')
    axs[0].axvline(x=n_samples-1, color='red', linestyle='--', linewidth=2, label='End')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title(f'Channel 7 (GAIT) - Gait Cycle (Samples {start_idx} to {end_idx})')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)
    
    axs[1].plot(time_axis, ch10_cycle, 'g-', linewidth=1)
    axs[1].axvline(x=0, color='green', linestyle='--', linewidth=2)
    axs[1].axvline(x=n_samples-1, color='red', linestyle='--', linewidth=2)
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Channel 10 (GL) - Gait Cycle')
    axs[1].grid(True, alpha=0.3)
    
    axs[2].plot(time_axis, ch13_cycle, 'r-', linewidth=1)
    axs[2].axvline(x=0, color='green', linestyle='--', linewidth=2)
    axs[2].axvline(x=n_samples-1, color='red', linestyle='--', linewidth=2)
    axs[2].set_ylabel('Amplitude')
    axs[2].set_xlabel('Sample (relative)')
    axs[2].set_title('Channel 13 (VL) - Gait Cycle')
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def process_emg(header_path, dat_path):

    gains, baselines = read_header(header_path)

    raw, p = read_raw_data(dat_path, gains, baselines)
    
    df = extract_first_samples(p)

    ch7_full = p[:, 7]
    ch10_full = p[:, 10]
    ch13_full = p[:, 13]

    start_idx, end_idx = zoom_and_segment(ch7_full, ch10_full, ch13_full)
    
    if start_idx is not None and end_idx is not None:
        print(f"Durasi: {end_idx - start_idx} samples")

        gait_cycle = p[start_idx:end_idx, 7]
        ch10_cycle = p[start_idx:end_idx, 10]
        ch13_cycle = p[start_idx:end_idx, 13]

        plot_segmented_gait_cycle(gait_cycle, ch10_cycle, ch13_cycle, start_idx, end_idx)
        
        return df, gait_cycle, ch10_cycle, ch13_cycle, (start_idx, end_idx)
    else:
        print("Segmentasi dibatalkan.")
        return df, None, None, None, None