from augmentation import augment
from augmentation import Augmentations
from model import train, test
from time import perf_counter

if __name__ == "__main__":
    for a in Augmentations:
        for i in range(0, 5):
            time_start = perf_counter()
            print(f"============== ITERATION {i} ==============")
            augment(a)
            train(f'{a}_{i}')
            test(f'{a}_{i}')
            time_end = perf_counter()
            time_duration = time_end - time_start
            with open('results/timings.txt', 'a') as f:
                f.write(f"Total execution time for Random Mask Factor {i}: {time_duration/60} minutes\n")