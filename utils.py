import librosa
import scipy.io.wavfile as wav
import librosa
import librosa.display as ld
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_spectrogram(wav_file, output_file):
    
    audio, sr = load_audio(wav_file)
    plt.specgram(audio, Fs=sr, cmap='magma')
    plt.colorbar(label='Intensity (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.savefig(output_file)
    plt.close()

def compare_spectrogram(wav_file1, wav_file2, output_file):

    audio1, sr1 = load_audio(wav_file1)
    audio2, sr2 = load_audio(wav_file2)
    D1 = librosa.amplitude_to_db(np.abs(librosa.stft(audio1)), ref=np.max)
    D2 = librosa.amplitude_to_db(np.abs(librosa.stft(audio2)), ref=np.max)
    
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 2, 1)
    ld.specshow(D1, sr=sr1, y_axis='linear', x_axis='time', cmap='magma') 
    plt.title('Spectrogram - File 1')

    plt.subplot(1, 2, 2)
    ld.specshow(D2, sr=sr2, y_axis='linear', x_axis='time', cmap='magma') 
    plt.title('Spectrogram - File 2')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_waveform(wav_file, output_file):

    audio, sr = load_audio(wav_file)
    
    time = np.arange(0, len(audio)) / sr
    
    plt.figure(figsize=(10, 4))
    
    plt.plot(time, audio)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def compare_waveform(wav_file1, wav_file2, output_file):

    audio1, sr1 = load_audio(wav_file1)
    audio2, sr2 = load_audio(wav_file2)
    
    time1 = np.arange(0, len(audio1)) / sr1
    time2 = np.arange(0, len(audio2)) / sr2
    
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(time1, audio1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform - File 1')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(time2, audio2)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform - File 2')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def mel_to_audio(mel_spectrogram, sr):

    #mel_spectrogram = librosa.db_to_power(mel_spectrogram)
    audio = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sr, n_fft=1024, hop_length=512)

    return audio

def audio_to_mel(file):
    audio , sr = load_audio(file)

    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=512, n_mels=128)

    return mel_spectrogram , sr

def load_audio(file_path):
    
    audio, sr = librosa.load(file_path, sr=None)

    return audio, sr

def save_audio(file_path, audio, sr):
    
    wav.write(file_path, sr, audio)

def get_result():
    file_path = os.path.join(os.getcwd(), 'results', 'demo')
    files = os.listdir(file_path)
    for f in files:
        if f.startswith("decision"):
            fp = os.path.join(file_path, f)
            df = pd.read_csv(fp)
            result = int(df.columns[1])
            break
    if result == 1:
        return "Anomaly"
    elif result == 0:
        return "Normal"