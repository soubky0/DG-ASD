from augmentation import main as augment
from model import train, test
from time import perf_counter

if __name__ == "__main__":
    for i in range(1, 5):
        time_start = perf_counter()
        print(f"============== ITERATION {i} ==============")
        augment(290)
        train(f'rerun_mask_factor_290_{i}')
        test(f'rerun_mask_factor_290_{i}')
        time_end = perf_counter()
        time_duration = time_end - time_start
        with open('results/timings.txt', 'a') as f:
            f.write(f"Total execution time for Random Mask Factor {i}: {time_duration/60} minutes\n")