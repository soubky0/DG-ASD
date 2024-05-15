from augmentation import *
from utils import *
from model import *

def main():
    normal_file = "audio/normal.wav"
    audio, sr = librosa.load(normal_file)
    augmented_audio = spec_augment(audio, sr)
    augmented_file = f"audio/spec_augmented.wav"
    wav.write(augmented_file, sr, augmented_audio)
    compare_spectrogram(normal_file, augmented_file, f"plots/spec_augment_spectrogram.png")
    compare_waveform(normal_file, augmented_file, f"plots/spec_augment_waveform.png")
    # for i in range(1, 10):
    #     mask_length = i / 100.0
    #     augmented_audio = random_time_mask(audio, mask_length)
    #     augmented_file = f"audio/augmented_{i}.wav"
    #     wav.write(augmented_file, sr, augmented_audio)
    #     compare_spectrogram(normal_file, augmented_file, f"plots/spectrogram_{i}.png")
    #     compare_waveform(normal_file, augmented_file, f"plots/waveform_{i}.png")

    
if __name__ == '__main__':
    main()
