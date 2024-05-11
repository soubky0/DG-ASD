from augmentation import main as augment
from train import main as train
from test import main as test

if __name__ == "__main__":
    for i in range(10,1000, 10):
        print(f"============== MASKING FACTOR {i} ==============")
        augment(i)
        train(f'masking_factor_{i}')
        test(f'masking_factor_{i}')