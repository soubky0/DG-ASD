from augmentation import augment, normal
from augmentation import Augmentations
from model import train, test
from time import perf_counter
from utils import post_train

def baseline():
    time_start = perf_counter()
    train('baseline')
    test('baseline')
    time_end = perf_counter()
    time_duration = time_end - time_start
    with open('results/timings.txt', 'a') as f:
        f.write(f"Total execution time of Baseline: {time_duration/60} minutes\n")

def apply_augmentation(a: Augmentations, all=False, **kwargs):
    if all:
        for ag in Augmentations:
            time_start = perf_counter()
            augment(ag, **kwargs)
            train(f'{ag.name}')
            test(f'{ag.name}')
            time_end = perf_counter()
            time_duration = time_end - time_start
            with open('results/timings.txt', 'a') as f:
                f.write(f"Total execution time of {ag.name}: {time_duration/60} minutes\n")
    else:
        time_start = perf_counter()
        augment(a, **kwargs)
        train(f'{a.name}')
        test(f'{a.name}')
        time_end = perf_counter()
        time_duration = time_end - time_start
        with open('results/timings.txt', 'a') as f:
            f.write(f"Total execution time of {a.name}: {time_duration/60} minutes\n")

if __name__ == "__main__":
    
    time_start = perf_counter()
    augment(Augmentations.TIME_MASK_RAW_2, mask_factor=320)
    train(f'time_mask_raw_old_320_2')
    test(f'time_mask_raw_old_320_2')
    time_end = perf_counter()
    time_duration = time_end - time_start
    with open('results/timings.txt', 'a') as f:
        f.write(f"Total execution time of time_mask_raw_2_320_2: {time_duration/60} minutes\n")
    post_train(f'time_mask_raw_old_320_2')
