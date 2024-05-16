from augmentation import *
from utils import *
from model import *

def main():
    audio, sr = load_audio('audio/normal.wav')
    augmented_audio = apply_augmentation(audio, sr, Augmentations.SPEC_AUGMENT)
    save_audio('audio/augmented.wav', augmented_audio, sr)
    compare_waveform('audio/normal.wav', 'audio/augmented.wav', 'plots/time_warp_waveform.png')
    compare_spectrogram('audio/normal.wav', 'audio/augmented.wav', 'plots/time_warp_spectogram.png')
    
if __name__ == '__main__':
    main()
