from augmentation import augment
from augmentation import Augmentations
from model import train, test
from time import perf_counter

def baseline():
  for i in range(1, 6):
        time_start = perf_counter()
        print(f"============== ITERATION {i} ==============")
        # augment(a)
        train(f'Baseline_{i}')
        test(f'Baseline_{i}')
        time_end = perf_counter()
        time_duration = time_end - time_start
        with open('results/baseLine_timings.txt', 'a') as f:
            f.write(f"Total execution time for Baseline {i}: {time_duration/60} minutes\n")

def all_augmentation():
    for a in Augmentations:
        time_start = perf_counter()
        print(f"============== AUGMENTATION {a.name} ==============")
        augment(a)
        train(f'{a.name}')
        test(f'{a.name}')
        time_end = perf_counter()
        time_duration = time_end - time_start
        with open('results/augmentation_timings.txt', 'a') as f:
            f.write(f"Total execution time for Random Mask Factor {i}: {time_duration/60} minutes\n")


if __name__ == "__main__":
    # baseline()
    # all_augmentation()
        
    for i in range(1,6):
        time_start = perf_counter()
        augment(Augmentations.TIME_MASK)
        train(f'TIME_MASK_320_iteration_{i}')
        test(f'TIME_MASK_320_iteration_{i}')
        time_end = perf_counter()
        time_duration = time_end - time_start
        with open('results/augmentation_timings.txt', 'a') as f:
            f.write(f"Total execution time for Random Mask Factor {i}: {time_duration/60} minutes\n")