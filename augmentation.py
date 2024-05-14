import os
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import shutil
from tqdm import tqdm

def random_time_mask(audio, mask_length=0.1, mask_value=0.0):
    """
    Applies random time masking to an audio signal.

    Args:
        audio (numpy.ndarray): The input audio signal.
        mask_length (float): The fraction of the audio to mask (0.0 to 1.0).
        mask_value (float): The value to use for masking (usually 0.0 for silence).

    Returns:
        numpy.ndarray: The masked audio signal.
    """
    
    # Ensure valid mask length
    if not 0.0 <= mask_length <= 1.0:
        raise ValueError("mask_length must be between 0.0 and 1.0")

    # Calculate mask start and end points
    num_samples = len(audio)
    mask_start = np.random.randint(0, num_samples - int(mask_length * num_samples))
    mask_end = mask_start + int(mask_length * num_samples)

    # Apply the mask
    masked_audio = audio.copy()
    masked_audio[mask_start:mask_end] = mask_value

    return masked_audio

def load_audio(file_path):
    """
    Load audio file.
    
    Parameters:
        file_path (str): Path to the input audio file.
    
    Returns:
        tuple: Tuple containing audio data and sample rate.
    """
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def save_audio(file_path, audio, sr):
    """
    Save audio file.
    
    Parameters:
        file_path (str): Path to save the audio file.
        audio (np.ndarray): Audio data.
        sr (int): Sample rate.
    """
    wavfile.write(file_path, sr, audio)

def augment_audio(input_dir, output_dir, mask_length=0.1):
    """
    Augment audio files in the input directory and save them to the output directory.
    
    Parameters:
        input_dir (str): Path to the input directory containing original audio files.
        output_dir (str): Path to the output directory to save augmented audio files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("============== BEGIN AUGMENTATION ==============")
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".wav"):
            input_file_path = os.path.join(input_dir, filename)
            audio, sr = load_audio(input_file_path)

            augmented_audio = random_time_mask(audio, mask_length)
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

def main(mask_length=0.1):
    try:
        shutil.rmtree(os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'train'))
        shutil.rmtree(os.path.join(os.getcwd(), 'dev_data', 'processed', 'gearbox'))
    except FileNotFoundError:
        pass
    os.makedirs(os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'train'))
    input_directory = os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'normal')
    output_directory = os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'train')

    augment_audio(input_directory, output_directory, mask_length)
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
