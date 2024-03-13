import torchaudio
import numpy as np
import matplotlib.pyplot as plt

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

superlet_result = superlets(waveform[0].numpy(), sample_rate, np.linspace(1, sample_rate // 2, 100), c1, orders)
superlet_spec = torchaudio.transforms.AmplitudeToDB()(np.abs(superlet_result), ref=np.max)

# Visualization
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

plot_spectrogram('SLT (Original)', superlet_spec, 0, time_extent)
#plot_spectrogram('SLT (PyTorch)', superlet_spec, 1, time_extent)

plt.tight_layout()
plt.savefig('result')