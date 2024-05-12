from augmentation import main as augment
from train import main as train
from test import main as test

from time import perf_counter

if __name__ == "__main__":
    for i in range(10,1000, 10):
        time_start = perf_counter()
        print(f"============== MASKING FACTOR {i} ==============")
        augment(i)
        train(f'masking_factor_{i}')
        test(f'masking_factor_{i}')
        time_end = perf_counter()
        time_duration = time_end - time_start
        with open('results/timings.txt', 'a') as f:
            f.write(f"Total execution time for Masking factor {i}: {time_duration/60} minutes\n")