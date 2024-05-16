from augmentation import *
from utils import *
from model import *

def main():
    audio, sr = load_audio('audio/normal.wav')
    time_warped = time_warp(audio, sr, 0.4)
    save_audio('audio/augmented.wav', time_warped, sr)
    compare_waveform('audio/normal.wav', 'audio/augmented.wav', 'plots/time_warp_waveform.png')
    compare_spectrogram('audio/normal.wav', 'audio/augmented.wav', 'plots/time_warp_spectogram.png')
    
if __name__ == '__main__':
    main()
