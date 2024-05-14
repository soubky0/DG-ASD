import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import os
import librosa
from augmentation import random_time_mask
def plot_spectrogram(wav_file, output_file):
    # Read the WAV file
    sample_rate, data = wav.read(wav_file)

    plt.specgram(data, Fs=sample_rate, cmap='magma')
    plt.colorbar(label='Intensity (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.savefig(output_file)
    plt.close()

import librosa
import librosa.display
import matplotlib.pyplot as plt

def compare_spectrogram(wav_file1, wav_file2, output_file):
    """
    Compares the spectrograms of two WAV files and saves the plot.
    """

    y1, sr1 = librosa.load(wav_file1)
    y2, sr2 = librosa.load(wav_file2)

    # Calculate spectrograms using librosa
    D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
    D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)

    # Plot spectrograms
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 2, 1)
    librosa.display.specshow(D1, sr=sr1, y_axis='linear', x_axis='time', cmap='magma') 
    plt.title('Spectrogram - File 1')

    plt.subplot(1, 2, 2)
    librosa.display.specshow(D2, sr=sr2, y_axis='linear', x_axis='time', cmap='magma') 
    plt.title('Spectrogram - File 2')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_waveform(wav_file, output_file):

    sample_rate, data = wav.read(wav_file)
    
    time = np.arange(0, len(data)) / sample_rate
    
    plt.figure(figsize=(10, 4))
    
    plt.plot(time, data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def compare_waveform(wav_file1, wav_file2, output_file):

    sample_rate1, data1 = wav.read(wav_file1)
    sample_rate2, data2 = wav.read(wav_file2)
    
    time1 = np.arange(0, len(data1)) / sample_rate1
    time2 = np.arange(0, len(data2)) / sample_rate2
    
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(time1, data1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform - File 1')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(time2, data2)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform - File 2')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    normal_file = "audio/normal.wav"
    audio, sr = librosa.load(normal_file)
    for i in range(1, 10):
        mask_length = i / 100.0
        augmented_audio = random_time_mask(audio, mask_length)
        augmented_file = f"audio/augmented_{i}.wav"
        wav.write(augmented_file, sr, augmented_audio)
        compare_spectrogram(normal_file, augmented_file, f"plots/spectrogram_{i}.png")
        compare_waveform(normal_file, augmented_file, f"plots/waveform_{i}.png")
    
if __name__ == '__main__':
    main()
