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
    
    for i in range(1, 6):
        time_start = perf_counter()
        normal()
        train(f'baseline_{i}')
        test(f'baseline_{i}')
        time_end = perf_counter()
        time_duration = time_end - time_start
        with open('results/timings.txt', 'a') as f:
            f.write(f"Total execution time of baseline_{i}: {time_duration/60} minutes\n")
            
    for i in [60, 220, 320]:
        for j in range(1,6):
            time_start = perf_counter()
            augment(Augmentations.TIME_MASK_RAW_2, mask_factor=i)
            train(f'time_mask_raw_old_{i}_{j}')
            test(f'time_mask_raw_old_{i}_{j}')
            time_end = perf_counter()
            time_duration = time_end - time_start
            with open('results/timings.txt', 'a') as f:
                f.write(f"Total execution time of time_mask_raw_2_{i}_{j}: {time_duration/60} minutes\n")
            post_train(f'time_mask_raw_old_{i}_{j}')
