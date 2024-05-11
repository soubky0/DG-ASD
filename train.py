import random
import numpy as np
import torch
from augmentation import main as augment

# original lib
import common as com
from networks.models import Models

########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################


def parse_args():
    parser = com.get_argparse()
    # read parameters from yaml
    flat_param = com.param_to_args_list(params=param)
    args = parser.parse_args(args=flat_param)

    args.cuda = args.use_cuda and torch.cuda.is_available()

    # Python random
    random.seed(args.seed)
    # Numpy
    np.random.seed(args.seed)
    # Pytorch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    return args


def main(tag=0):
    args = parse_args()
    args.tag = tag
    print(args)

    net = Models(args.model).net(args=args, train=True, test=False)

    print("============== BEGIN TRAIN ==============")
    for epoch in range(1, args.epochs + 2):
        net.train(epoch)
    print("============ END OF TRAIN ============")


if __name__ == "__main__":
    for i in range(10,1000):
        print(f"============== MASKING FACTOR {i} ==============")
        augment(i)
        main(i)