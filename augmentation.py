from audiomentations import TimeMask
from scipy.io import wavfile as wav
import numpy as np
import os

dir = "/home/omar/Uni/bachelor/my-src/dev_data/raw/gearbox/"
input_dir = dir + "normal"
output_dir = dir +"augmented"
final_dir = dir + "train"

transform = TimeMask(
    min_band_part=0.1,
    max_band_part=0.15,
    fade=True,
    p=1.0,
)

for filename in os.listdir(input_dir):
    f = os.path.join(input_dir, filename)
    wavfile = wav.read(f)
    a = np.array(wavfile[1], dtype="float32")
    augmented_sound = transform(a, sample_rate=16000)
    filename = filename.split(".")[0] + "_augmented.wav"
    f = os.path.join(output_dir, filename)
    wav.write(f, 16000, augmented_sound)

for f in os.listdir(input_dir):
    src_path = os.path.join(input_dir, f)
    dst_path = os.path.join(final_dir, f)
    os.rename(src_path, dst_path)
    
for f in os.listdir(output_dir):
    src_path = os.path.join(output_dir, f)
    dst_path = os.path.join(final_dir, f)
    os.rename(src_path, dst_path)
