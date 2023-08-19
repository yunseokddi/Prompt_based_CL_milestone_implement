import torch
import torch.backends.cudnn as cudnn
import warnings
import argparse
import numpy as np
import random

from parse_config import get_args_parser
from pathlib import Path
from tensorboard_logger import configure
from utils.utils import init_distributed_mode
from data_loader.data_loaders import DataLoader

warnings.filterwarnings('ignore')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def main(args):
    init_distributed_mode(args)

    device = torch.device(args.device)

    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_loader = DataLoader(args)

    train_loader, val_loader = data_loader.get_dataloader()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VPT training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == "CIFAR_VPT":
        config_parser = subparser.add_parser('CIFAR_VPT', help='CUB2000 Visual Prompt Tuning')
        get_args_parser(config_parser)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.tensorboard:
        configure("runs/%s" % (args.dataset))

    main(args)

'''
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env train.py \
        CIFAR_VPT \
        --model vit_base_patch16_224 \
        --batch-size 128 \
        --tensorboard True
'''