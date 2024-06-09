import os
import numpy as np
import shutil
from tqdm import tqdm
import random
from utils import *
from enum import Enum
from audiomentations import Compose, TimeMask

class Augmentations(Enum):
    TIME_MASK_LIBRARY = "time_mask_audiomentations"
    TIME_MASK = "time_mask"
    FREQUENCY_MASK = "frequency_mask"
    TIME_WARP = "time_warp"
    SPEC_AUGMENT = "spec_augment"

def time_mask(audio, sr):
    masked_audio = audio.copy()
    mask_length = int(len(audio) / 320) #TODO
    start = np.random.randint(0, len(audio) - mask_length)
    masked_audio[start:start+mask_length] = 0
    return masked_audio


def time_mask_audiomentations(audio, sr):
    
    audioment = Compose([TimeMask(min_band_part=0.1, max_band_part=0.2, p=1.0)])
    audio = audioment(audio, sr)
    
    return audio

def freq_mask(audio, sr):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    num_mel_bins = mel_spec.shape[0]
    mask_size = int(random.uniform(0.1, 0.2) * num_mel_bins)
    mask_start = random.randint(0, num_mel_bins - mask_size)
    
    mel_spec[mask_start:mask_start + mask_size, :] = 0

    masked_audio = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sr)

    if len(masked_audio) < len(audio):
        masked_audio = np.pad(masked_audio, (0, len(audio) - len(masked_audio)), 'constant')
    return masked_audio

def time_warp(audio, sr):

    # Convert the audio to a mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    
    # Create a time axis
    time_steps = np.arange(mel_spec.shape[1])
    
    # Generate a random warp for each time step
    random_warp = np.random.uniform(-0.1, 0.1, mel_spec.shape[1])
    time_steps_warped = time_steps + random_warp
    
    # Ensure the warped time steps are within valid bounds
    time_steps_warped = np.clip(time_steps_warped, 0, mel_spec.shape[1] - 1)
    
    # Interpolate the warped spectrogram
    mel_spec_warped = np.zeros_like(mel_spec)
    for i in range(mel_spec.shape[0]):
        mel_spec_warped[i] = np.interp(time_steps, time_steps_warped, mel_spec[i])
    
    # Convert the warped mel-spectrogram back to audio
    warped_audio = librosa.feature.inverse.mel_to_audio(mel_spec_warped, sr=sr)

    if len(warped_audio) < len(audio):
        warped_audio = np.pad(warped_audio, (0, len(audio) - len(warped_audio)), 'constant')

    return warped_audio

def spec_augment(audio, sr):

    audio = time_mask(audio, sr)
    audio = freq_mask(audio, sr)
    audio = time_warp(audio, sr)

    return audio

def apply_augmentation(audio, sr, augmentation):    
    augmentation_functions = {
        Augmentations.TIME_MASK: time_mask,
        Augmentations.FREQUENCY_MASK: freq_mask,
        Augmentations.TIME_WARP: time_warp,
        Augmentations.SPEC_AUGMENT: spec_augment,
        Augmentations.TIME_MASK_LIBRARY: time_mask_audiomentations
    }
    
    if augmentation not in augmentation_functions:
        raise ValueError("Invalid augmentation technique specified.")
    
    return augmentation_functions[augmentation](audio, sr)

def augment(augmentation: Augmentations):
    
    normal_dir = os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'normal')
    augmented_dir = os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'augmented')
    train_dir = os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'train')
    processed_dir = os.path.join(os.getcwd(), 'dev_data', 'processed', 'gearbox')

    try:
        shutil.rmtree(processed_dir)
    except:
        pass
    
    print(f"============== BEGIN {augmentation.name} AUGMENTATION ==============")

    for filename in tqdm(os.listdir(normal_dir)):
        file = os.path.join(normal_dir, filename)
        audio, sr = load_audio(file)
        augmented_audio = apply_augmentation(audio, sr, augmentation)
        parts = filename.split("_")

        count = parts[5]
        is_target = "target" in filename

        if is_target:
            new_count = str(int(count) + 10)
        else:
            new_count = str(int(count) + 1000)

        augmented_filename = filename.replace(f"_{count}_", f"_{new_count}_")
        augmented_file = os.path.join(augmented_dir, augmented_filename)
        save_audio(augmented_file, augmented_audio, sr)
    copy_files(augmented_dir, train_dir)
    copy_files(normal_dir, train_dir)
    print("============== END OF AUGMENTATION ==============")

