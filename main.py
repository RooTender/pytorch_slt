import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa

from python import superlet as slt_orig
import superlet as slt

filename = 'wav/03a01Fa.wav'

def plot_spectrogram(title: str, data: any, x_pos: int, time_extent: any):
    axs[x_pos].imshow(data, aspect='auto', origin='lower', extent=time_extent)
    axs[x_pos].set_ylabel('Frequency [Hz]')
    axs[x_pos].set_xlabel('Time [sec]')
    axs[x_pos].set_title(title, loc='left')

# PyTorch
waveform, sample_rate = torchaudio.load(filename)
time_extent = [0, waveform.size(1) / sample_rate, 0, sample_rate / 2]

# Superlet Transform
c1 = 2  # Base number of cycles
orders = (5, 10)  # Order range

superlet_result = slt_orig.superlets(waveform[0].numpy(), sample_rate, np.linspace(1, sample_rate // 2, 100), c1, orders)
superlet_spec_orig = slt_orig.amplitude_to_db(np.abs(superlet_result))

superlet_result = slt.superlets(waveform, sample_rate, torch.linspace(1, sample_rate // 2, 100), c1, orders)
superlet_spec = slt.amplitude_to_db(superlet_result.abs())

# Visualization
fig, axs = plt.subplots(2, 1, figsize=(20, 10))

#plot_spectrogram('STFT (Librosa)', spec_librosa_db, 0, time_extent)
plot_spectrogram('SLT (Original)', superlet_spec_orig, 0, time_extent)
plot_spectrogram('SLT (PyTorch)', superlet_spec, 1, time_extent)

plt.tight_layout()
plt.savefig('result')