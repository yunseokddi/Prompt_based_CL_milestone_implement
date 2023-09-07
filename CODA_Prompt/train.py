import argparse
import warnings
import torch.backends.cudnn
import sys
import random
import numpy as np

from pathlib import Path
from parse_config import CIFAR100_get_args_parser
from tensorboard_logger import configure
from utils.utils import init_distributed_mode
from data_loader.data_loaders import ContinualDataLoader

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

    CL_dataloader = ContinualDataLoader(args)

    data_loader, class_mask = CL_dataloader.get_dataloader()




if __name__ == "__main__":
    parser = argparse.ArgumentParser('CODA-Prompt training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_coda_prompt':
        config_parser = subparser.add_parser('cifar100_coda_prompt', help='Split-CIFAR100 CODA-Prompt configs')
        CIFAR100_get_args_parser(config_parser)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.tensorboard:
        configure("runs/%s" % (args.dataset))

    main(args)

    sys.exit(0)



'''
CUDA_VISIBLE_DEVICES=2,3 torchrun \
        --nproc_per_node=2 \
        train.py \
        cifar100_coda_prompt \
        --model vit_base_patch16_224 \
        --batch-size 128 \
        --tensorboard True
        
CUDA_VISIBLE_DEVICES=0 torchrun \
        --nproc_per_node=1 \
        train.py \
        cifar100_coda_prompt \
        --model vit_base_patch16_224 \
        --batch-size 128 \
        --tensorboard True
        
python3 train.py \
        cifar100_coda_prompt \
        --model vit_base_patch16_224 \
        --batch-size 128 \
        --tensorboard True
'''
