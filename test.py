import random
import numpy as np
import torch

# original lib
import common as com
from networks.models import Models

########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################


def parse_args(tag, demo=False):
    parser = com.get_argparse()
    # read parameters from yaml
    flat_param = com.param_to_args_list(params=param)
    flat_param.extend(["-tag", tag])
    args = parser.parse_args(args=flat_param)
    args.demo = demo
    # read parameters from command line
    args = parser.parse_args(namespace=args)

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
    args = parse_args(tag)
    print(args)

    net = Models(args.model).net(args=args, train=False, test=True)

    net.test()


def model_test():
    args = parse_args()
    print(args, True)

    net = Models(args.model).net(args=args, train=False, test=True)

    return net.test()


if __name__ == "__main__":
    main()
