import os
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import shutil
from tqdm import tqdm

def time_masking(audio, mask_factor=10):
    """
    Apply time masking to the audio.
    
    Parameters:
        audio (np.ndarray): Input audio array.
        mask_factor (int): Factor to determine the length of the mask.
    
    Returns:
        np.ndarray: Audio with time masking applied.
    """
    masked_audio = audio.copy()
    # Determine mask length
    mask_length = int(len(audio) / mask_factor)
    # Randomly select a starting point for the mask
    start = np.random.randint(0, len(audio) - mask_length)
    # Apply mask
    masked_audio[start:start+mask_length] = 0
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

def augment_audio(input_dir, output_dir):
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

            # Load audio
            audio, sr = load_audio(input_file_path)

            # Apply time masking augmentation
            augmented_audio = time_masking(audio)

            # Generate output file path
            output_file_path = os.path.join(output_dir, filename.replace("_normal_", "_anomaly_"))

            # Save augmented audio
            save_audio(output_file_path, augmented_audio, sr)

            # Copy original file to output directory
            shutil.copy2(input_file_path, output_dir)

    print("============== END OF AUGMENTATION ==============")

def main():
    try:
        shutil.rmtree(os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'train'))
    except FileNotFoundError:
        pass
    os.makedirs(os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'train'))
    input_directory = os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'normal')
    output_directory = os.path.join(os.getcwd(), 'dev_data', 'raw', 'gearbox', 'train')
    
    augment_audio(input_directory, output_directory)

if __name__ == "__main__":
    main()
