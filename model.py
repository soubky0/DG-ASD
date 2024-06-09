import random
import numpy as np
import torch
import common as com
from networks.models import Models
from utils import *
param = com.yaml_load()

def parse_args(tag, score="MSE"):
    parser = com.get_argparse()
    flat_param = com.param_to_args_list(params=param)
    flat_param.extend(["-tag", tag])
    flat_param.extend(["--score", score])
    args = parser.parse_args(args=flat_param)
    args = parser.parse_args(namespace=args)
    args.cuda = args.use_cuda and torch.cuda.is_available()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    return args

def demo(model_name):
    
    test(model_name, demo=True)
    result = get_result()
    return result


def test(tag=0):
    args = parse_args(tag)
    print(args)

    net = Models(args.model).net(args=args, train=False, test=True)
    net.test()
    args = parse_args(tag, "MAHALA")
    net = Models(args.model).net(args=args, train=False, test=True)
    net.test()

def train(tag=0):
    args = parse_args(tag)
    print(args)

    net = Models(args.model).net(args=args, train=True, test=False)

    print("============== BEGIN TRAIN ==============")
    for epoch in range(1, args.epochs + 2):
        net.train(epoch)
    print("============ END OF TRAIN ============")