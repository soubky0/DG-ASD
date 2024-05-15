from augmentation import main as augment
from model import train, test
from time import perf_counter

if __name__ == "__main__":
    for i in range(1, 10):
        mask_length = i / 100.0
        time_start = perf_counter()
        print(f"============== MASKING FACTOR {i} ==============")
        augment(mask_length)
        train(f'random_mask_factor_{mask_length}')
        test(f'random_mask_factor_{mask_length}')
        time_end = perf_counter()
        time_duration = time_end - time_start
        with open('results/timings.txt', 'a') as f:
            f.write(f"Total execution time for Random Mask Factor {i}: {time_duration/60} minutes\n")