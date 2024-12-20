import os
import numpy as np
import shutil
from tqdm import tqdm
import random
from utils import *
from enum import Enum
from audiomentations import Compose, TimeMask
from sklearn.model_selection import train_test_split

class Augmentations(Enum):
    TIME_MASK_RAW = "time_mask_raw"
    TIME_MASK_RAW_2 = "time_mask_raw_2"
    TIME_MASK_SPEC = "time_mask_spec"
    FREQUENCY_MASK = "frequency_mask"
    TIME_MASK_AUDIOMENTATIONS = "time_mask_audiomentations"
    TIME_WARP = "time_warp"
    SPEC_AUGMENT = "spec_augment"

def time_mask_raw_2(audio, sr, mask_factor=2):

    masked_audio = audio.copy()
    mask_length = int(len(audio) / mask_factor)
    start = np.random.randint(0, len(audio) - mask_length)
    masked_audio[start:start+mask_length] = 0

    return masked_audio

def time_mask_spec(audio, sr, T=20, num_masks=1):
    S_amp = librosa.stft(audio)
    S = librosa.amplitude_to_db(np.abs(S_amp), ref=np.max)
    num_time_steps = S.shape[1]

    for _ in range(num_masks):
        t = np.random.uniform(low=0, high=T)
        t0 = np.random.uniform(low=0, high=num_time_steps - t)
        S[:, int(t0):int(t0 + t)] = 0

    S_masked_amplitude = librosa.db_to_amplitude(S)
    audio_masked = librosa.istft(S_masked_amplitude, length=len(audio))
    audio_masked = librosa.util.normalize(audio_masked)

    return audio_masked

def time_mask_raw(audio, sr, min_T=50, max_T=500, num_masks=1):
    masked_audio = audio.copy()
    num_samples = len(masked_audio)
    
    for _ in range(num_masks):
        mask_len = int(np.random.uniform(min_T, max_T) / 1000 * sr)
        t0 = np.random.randint(0, num_samples - mask_len)
        masked_audio[t0:t0 + mask_len] = 0

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

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    time_steps = np.arange(mel_spec.shape[1])
    random_warp = np.random.uniform(-0.1, 0.1, mel_spec.shape[1])
    time_steps_warped = time_steps + random_warp
    time_steps_warped = np.clip(time_steps_warped, 0, mel_spec.shape[1] - 1)
    mel_spec_warped = np.zeros_like(mel_spec)
    
    for i in range(mel_spec.shape[0]):
        mel_spec_warped[i] = np.interp(time_steps, time_steps_warped, mel_spec[i])
    
    warped_audio = librosa.feature.inverse.mel_to_audio(mel_spec_warped, sr=sr)

    if len(warped_audio) < len(audio):
        warped_audio = np.pad(warped_audio, (0, len(audio) - len(warped_audio)), 'constant')

    return warped_audio

def spec_augment(audio, sr):

    audio = time_mask_spec(audio, sr)
    audio = freq_mask(audio, sr)
    audio = time_warp(audio, sr)

    return audio

def apply_augmentation(audio, sr, augmentation, **kwargs):    
    augmentation_functions = {
        Augmentations.TIME_MASK_RAW: time_mask_raw,
        Augmentations.TIME_MASK_SPEC: time_mask_spec,
        Augmentations.FREQUENCY_MASK: freq_mask,
        Augmentations.TIME_MASK_AUDIOMENTATIONS: time_mask_audiomentations,
        Augmentations.TIME_WARP: time_warp,
        Augmentations.SPEC_AUGMENT: spec_augment,
        Augmentations.TIME_MASK_RAW_2: time_mask_raw_2,
    }
    
    if augmentation not in augmentation_functions:
        raise ValueError("Invalid augmentation technique specified.")
    
    return augmentation_functions[augmentation](audio, sr, **kwargs)

def augment(augmentation: Augmentations, **kwargs):
    
    normal_dir = os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'normal')
    augmented_dir = os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'augmented')
    validation_dir = os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'validation')
    train_dir = os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'train')
    processed_dir = os.path.join(os.getcwd(), 'dev_data', 'processed', 'gearbox')

    try:
        shutil.rmtree(augmented_dir)
        shutil.rmtree(validation_dir)
        shutil.rmtree(train_dir)
        shutil.rmtree(processed_dir)
    except:
        pass
    
    os.makedirs(train_dir)
    os.makedirs(augmented_dir)
    os.makedirs(validation_dir)
    
    config = load_config()
    validation_split =  config.get('--validation_split', 0.1)

    filenames = os.listdir(normal_dir)
    train_filenames, val_filenames = train_test_split(filenames, test_size=validation_split)
    
    print(f"============== BEGIN {augmentation.name} AUGMENTATION ==============")

    for filename in tqdm(train_filenames):
        file = os.path.join(normal_dir, filename)
        audio, sr = load_audio(file)
        augmented_audio = apply_augmentation(audio, sr, augmentation, **kwargs)
        parts = filename.split("_")

        count = parts[5]
        is_target = "target" in filename

        if is_target:
            new_count = str(int(count) + 10)
        else:
            new_count = str(int(count) + 1000)

        augmented_filename = filename.replace(f"_{count}_", f"_{new_count}_augmented_")
        augmented_file = os.path.join(augmented_dir, augmented_filename)
        save_audio(augmented_file, augmented_audio, sr)
    copy_files(augmented_dir, train_dir)

    for filename in train_filenames:
        file = os.path.join(normal_dir, filename)
        dest_file = os.path.join(train_dir, filename)
        shutil.copy(file, dest_file)

    for filename in val_filenames:
        file = os.path.join(normal_dir, filename)
        dest_file = os.path.join(validation_dir, filename)
        shutil.copy(file, dest_file)
    
    print("============== END OF AUGMENTATION ==============")

def normal():
    normal_dir = os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'normal')
    validation_dir = os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'validation')
    train_dir = os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'train')
    processed_dir = os.path.join(os.getcwd(), 'dev_data', 'processed', 'gearbox')

    try:
        shutil.rmtree(validation_dir)
        shutil.rmtree(train_dir)
        shutil.rmtree(processed_dir)
    except:
        pass
    
    os.makedirs(train_dir)
    os.makedirs(validation_dir)
    
    config = load_config()
    validation_split =  config.get('--validation_split', 0.1)

    filenames = os.listdir(normal_dir)
    train_filenames, val_filenames = train_test_split(filenames, test_size=validation_split)

    for filename in train_filenames:
        file = os.path.join(normal_dir, filename)
        dest_file = os.path.join(train_dir, filename)
        shutil.copy(file, dest_file)

    for filename in val_filenames:
        file = os.path.join(normal_dir, filename)
        dest_file = os.path.join(validation_dir, filename)
        shutil.copy(file, dest_file)