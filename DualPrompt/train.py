import sys
import warnings
import argparse
import torch
import torch.backends.cudnn
import random
import numpy as np


from parse_config import CIFAR100_get_args_parser
from pathlib import Path
from utils.utils import init_distributed_mode
from data_loader.data_loaders import ContinualDataLoader
from timm.models import create_model

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

    print("Creating original model: {}".format(args.model))

    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_dualprompt':
        config_parser = subparser.add_parser('cifar100_dualprompt', help='Split-CIFAR100 DualPrompt configs')
        CIFAR100_get_args_parser(config_parser)

    else:
        assert "Check dataset"

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

    sys.exit(0)

'''
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --use_env train.py \
        cifar100_dualprompt \
        --model vit_base_patch16_224 \
        --batch-size 128

CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py cifar100_dualprompt --eval
'''