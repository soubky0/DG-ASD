from augmentation import augment
from augmentation import Augmentations
from model import train, test
from time import perf_counter
from utils import rename_directory

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
    for i in range(1,6):
        for j in range(1,6):
            time_start = perf_counter()
            augment(Augmentations.TIME_MASK_SPEC, T=j*10, num_masks=i)
            train(f'time_mask_spec_{i}_{j*10}')
            test(f'time_mask_spec_{i}_{j*10}')
            time_end = perf_counter()
            time_duration = time_end - time_start
            with open('results/timings.txt', 'a') as f:
                f.write(f"Total execution time of time_mask_spec_{i}_{j*10}: {time_duration/60} minutes\n")
