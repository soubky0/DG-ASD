import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

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

def compare_spectrogram(wav_file1, wav_file2, output_file):
    sample_rate1, data1 = wav.read(wav_file1)
    sample_rate2, data2 = wav.read(wav_file2)
    
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 2, 1)
    plt.specgram(data1, Fs=sample_rate1, cmap='magma')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram - File 1')
    
    plt.subplot(1, 2, 2)
    plt.specgram(data2, Fs=sample_rate2, cmap='magma')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
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

if __name__ == '__main__':
    plot_waveform('test.wav', 'waveform.png')
    plot_spectrogram('test.wav', 'spectrogram.png')
    compare_waveform('test1.wav', 'test2.wav', 'compare_waveform.png')
    compare_spectrogram('test1.wav', 'test2.wav', 'compare_spectrogram.png')
