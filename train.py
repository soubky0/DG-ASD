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

def main():
    parser = com.get_argparse()
    # read parameters from yaml
    flat_param = com.param_to_args_list(params=param)
    args = parser.parse_args(args=flat_param)
    # read parameters from command line
    args = parser.parse_args(namespace=args)
    print(args)

    train = True
    test = False
    
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

    net = Models(args.model).net(
        args=args,
        train=train,
        test=test
    )


    print(args.model)

    print("============== BEGIN TRAIN ==============")
    if train:
        for epoch in range(1, args.epochs + 2):
            net.train(epoch)
    print("============ END OF TRAIN ============")
    
    if test:
        net.test()

if __name__ == "__main__":
    main()