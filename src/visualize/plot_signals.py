import os
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

def load_ecg_mat(mat_file):
    mat_data = scipy.io.loadmat(mat_file)
    if 'val' not in mat_data:
        raise ValueError(f"Không tìm thấy key 'val' trong {mat_file}")
    return mat_data['val']  # shape (12, N)

def plot_ecg_signals(signals, title=None, save_path=None):
    fig, axes = plt.subplots(6, 2, figsize=(15, 12), sharex=True)
    axes = axes.flatten()
    for i, lead in enumerate(LEADS):
        axes[i].plot(signals[i], linewidth=0.7)
        axes[i].set_title(lead)
        axes[i].grid(True)
    plt.tight_layout()
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

def plot_spectrogram(signals, fs=500, lead_index=0, save_path=None):
    f, t, Sxx = spectrogram(signals[lead_index], fs)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
    plt.title(f"Spectrogram - Lead {LEADS[lead_index]}")
    plt.ylabel('Tần số [Hz]')
    plt.xlabel('Thời gian [s]')
    plt.colorbar(label='Công suất [dB]')
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

def plot_compare(original, denoised, lead_idx=0, fs=500):
    t = np.arange(original.shape[1]) / fs
    plt.figure(figsize=(12,4))
    plt.plot(t, original[lead_idx], label='Raw (mV)', linewidth=0.8)
    plt.plot(t, denoised[lead_idx], label='Denoised (mV)', linewidth=0.8)
    plt.legend()
    plt.title(f'Lead {LEADS[lead_idx]}')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.show()
