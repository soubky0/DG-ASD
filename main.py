import sys
import os
from time import perf_counter
from augmentation import augment, Augmentations
from model import train, test
from utils import post_train
import argparse

def main(mask_factor):
    time_start = perf_counter()
    augment(Augmentations.TIME_MASK_RAW_2, mask_factor=mask_factor)
    train(f'time_mask_{mask_factor}', mask_factor=mask_factor)
    test(f'time_mask_{mask_factor}', mask_factor=mask_factor)
    time_end = perf_counter()
    time_duration = time_end - time_start

    with open('results/timings.txt', 'a') as f:
        f.write(f"Total execution time of time_mask_{mask_factor}: {time_duration/60} minutes\n")

    post_train(f'time_mask_{mask_factor}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run augmentation and training")
    parser.add_argument("--mask_factor", type=int, required=True, help="Mask factor for augmentation")
    args = parser.parse_args()

    main(args.mask_factor)
