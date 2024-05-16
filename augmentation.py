import os
import numpy as np
import shutil
from tqdm import tqdm
from utils import *

from audiomentations import Compose, TimeMask

time_mask = Compose([TimeMask(min_band_part=0.1, max_band_part=0.2, p=1.0)])

def time_warp(audio, sample_rate, warp_factor=0.1):

    # Convert the audio to a mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    
    # Create a time axis
    time_steps = np.arange(mel_spec.shape[1])
    
    # Generate a random warp for each time step
    random_warp = np.random.uniform(-warp_factor, warp_factor, mel_spec.shape[1])
    time_steps_warped = time_steps + random_warp
    
    # Ensure the warped time steps are within valid bounds
    time_steps_warped = np.clip(time_steps_warped, 0, mel_spec.shape[1] - 1)
    
    # Interpolate the warped spectrogram
    mel_spec_warped = np.zeros_like(mel_spec)
    for i in range(mel_spec.shape[0]):
        mel_spec_warped[i] = np.interp(time_steps, time_steps_warped, mel_spec[i])
    
    # Convert the warped mel-spectrogram back to audio
    warped_audio = librosa.feature.inverse.mel_to_audio(mel_spec_warped, sr=sample_rate)

    if len(warped_audio) < len(audio):
        warped_audio = np.pad(warped_audio, (0, len(audio) - len(warped_audio)), 'constant')
        
    return warped_audio

def augment_audio(input_dir, output_dir, factor):

    print("============== BEGIN AUGMENTATION ==============")

    for filename in tqdm(os.listdir(input_dir)):
        file = os.path.join(input_dir, filename)
        audio, sr = audio_to_mel(file)
        augmented_audio = time_mask(audio, sr)
        augmented_audio = mel_to_audio(augmented_audio, sr) 
        parts = filename.split("_")

        count = parts[5]
        is_target = "target" in filename

        if is_target:
            new_count = str(int(count) + 1990)
        else:
            new_count = str(int(count) + 1000)

        filename = filename.replace(f"_{count}_", f"_{new_count}_")
        output_file_path = os.path.join(output_dir, filename)
        save_audio(output_file_path, augmented_audio, sr)
        shutil.copy2(file, output_dir)

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
