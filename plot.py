import matplotlib.pyplot as plt
import librosa

def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

def plot_waveform(waveform):
    plt.figure(figsize=(10, 4))
    plt.plot(waveform.t().numpy())
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()