import os
import numpy as np
import shutil
from tqdm import tqdm
from utils import *

def adaptive_time_mask(spec, threshold_factor=0.5, min_mask_length=10, max_mask_length=80):
    masked_spec = spec.copy()
    frame_energies = np.mean(spec, axis=0)  
    threshold = threshold_factor * np.mean(frame_energies)  
    mask_candidates = np.where(frame_energies < threshold)[0]

    if len(mask_candidates) > max_mask_length: 
        start_idx = np.random.choice(mask_candidates[:-max_mask_length])  
        mask_length = np.random.randint(min_mask_length, max_mask_length + 1)  
        masked_spec[:, start_idx : start_idx + mask_length] = 0

    return masked_spec

def time_warp(spec, W=80):

    spec_len = spec.shape[1]
    pt = np.random.randint(W, spec_len - W)
    w = np.random.randint(-W, W)
    warped_spec = np.zeros_like(spec)

    if w < 0:
        warped_spec[:, 0:pt + w] = spec[:, 0:pt + w]
        warped_spec[:, pt + w:spec_len] = spec[:, pt:spec_len]
    else:
        warped_spec[:, 0:pt] = spec[:, 0:pt]
        warped_spec[:, pt:spec_len - w] = spec[:, pt:spec_len - w] 
        warped_spec[:, spec_len - w:] = spec[:, -w:]  

    return warped_spec

def time_mask(spec, T=40, num_masks=2):

    for _ in range(num_masks):
        t = np.random.randint(low=0, high=spec.shape[1] - T)
        spec[:, t:t + T] = 0

    return spec

def freq_mask(spec, F=30, num_masks=2):
    
    for _ in range(num_masks):
        f = np.random.randint(low=0, high=spec.shape[0] - F)
        spec[f:f + F, :] = 0

    return spec

def spec_augment(audio, sr, W=80, T=40, F=30, num_time_masks=2, num_freq_masks=2):

    spec = audio_to_mel(audio, sr)
    spec = time_warp(spec, W=W)
    spec = time_mask(spec, T=T, num_masks=num_time_masks)
    spec = freq_mask(spec, F=F, num_masks=num_freq_masks)
    spec = mel_to_audio(spec, sr)
    
    return spec

def random_time_mask(audio, mask_length=0.1, mask_value=0.0):
    
    if not 0.0 <= mask_length <= 1.0:
        raise ValueError("mask_length must be between 0.0 and 1.0")

    num_samples = len(audio)
    mask_start = np.random.randint(0, num_samples - int(mask_length * num_samples))
    mask_end = mask_start + int(mask_length * num_samples)
    masked_audio = audio.copy()
    masked_audio[mask_start:mask_end] = mask_value

    return masked_audio

def augment_audio(input_dir, output_dir, factor):

    print("============== BEGIN AUGMENTATION ==============")

    for filename in tqdm(os.listdir(input_dir)):
        input_file_path = os.path.join(input_dir, filename)
        audio, sr = load_audio(input_file_path)
        # audio = audio_to_mel(audio, sr)
        # augmented_audio = adaptive_time_mask(audio, factor)
        parts = filename.split("_")
        count = parts[5]
        is_target = "_target_" in filename

        if is_target:
            filename = filename.replace("_target_", "_source_")
            new_count = str(int(count) + 1990)
        else:
            new_count = str(int(count) + 1000)

        filename = filename.replace(f"_{count}_", f"_{new_count}_")
        output_file_path = os.path.join(output_dir, filename)
        save_audio(output_file_path, augmented_audio, sr)
        shutil.copy2(input_file_path, output_dir)

    print("============== END OF AUGMENTATION ==============")

def main(factor=1):

    try:
        shutil.rmtree(os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'train'))
        shutil.rmtree(os.path.join(os.getcwd(), 'dev_data', 'processed', 'gearbox'))
    except FileNotFoundError:
        pass

    os.makedirs(os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'train'))
    input_directory = os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'normal')
    output_directory = os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'train')
    augment_audio(input_directory, output_directory, factor)
    source = 0
    target = 0

    for filename in os.listdir(output_directory):
        if "_source_" in os.path.join(output_directory, filename):
            source += 1
        elif "_target_" in os.path.join(output_directory, filename):
            target += 1

    print(f"Source: {source}, Target: {target}")

if __name__ == "__main__":
    main()
