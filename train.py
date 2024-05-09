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

def main():
    args = parse_args()

    print(args)

    net = Models(args.model).net(
        args=args,
        train=True,
        test=False
    )

    print("============== BEGIN TRAIN ==============")
    for epoch in range(1, args.epochs + 2):
        net.train(epoch)
    print("============ END OF TRAIN ============")
    
if __name__ == "__main__":
    main()