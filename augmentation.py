import os
import numpy as np
import shutil
from tqdm import tqdm
from utils import *

def random_time_mask(spec, max_masks=2, max_time_mask=10):
    """Applies random time masking to a spectrogram."""
    spec = spec.copy()  # Create a copy to avoid modifying the original
    num_masks = np.random.randint(1, max_masks + 1)  # Randomly choose the number of masks
    for _ in range(num_masks):
        t = np.random.randint(0, spec.shape[1] - max_time_mask)  # Random start time
        mask_length = np.random.randint(1, max_time_mask + 1)  # Random mask length
        spec[:, t : t + mask_length] = 0  # Mask the time segment
    return spec


def augment_audio(input_dir, output_dir, factor):

    print("============== BEGIN AUGMENTATION ==============")

    for filename in tqdm(os.listdir(input_dir)):
        file = os.path.join(input_dir, filename)
        audio, sr = audio_to_mel(file)
        augmented_audio = random_time_mask(audio, factor)
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
