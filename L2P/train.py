import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

import warnings

from parse_config import CIFAR100_get_args_parser
from pathlib import Path

warnings.filterwarnings('ignore')

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('L2P training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_l2p':
        config_parser = subparser.add_parser('cifar100_l2p', help='Split-CIFAR100 L2P configs')
        CIFAR100_get_args_parser(config_parser)

    else:
        assert "Check your param."

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

    sys.exit(0)