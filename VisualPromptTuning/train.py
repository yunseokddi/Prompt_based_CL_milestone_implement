import torch
import torch.backends.cudnn as cudnn
import warnings
import argparse
import numpy as np
import random
import os
import torchsummary

from parse_config import get_args_parser
from pathlib import Path
from tensorboard_logger import configure
from utils.utils import init_distributed_mode
from data_loader.data_loaders import DataLoader
from model.model import build_model
from trainer.optimizer import create_optimizer
from trainer.scheduler import create_scheduler
from trainer.losses import create_loss
from trainer.trainer import Trainer

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

    model = build_model(args)
    # print(model)

    criterion = create_loss(args)

    optimizer = create_optimizer(model, args)
    lr_scheduler = create_scheduler(optimizer, args)

    trainer = Trainer(model, criterion, data_loader, optimizer, lr_scheduler, device, args)

    trainer.train()

    print("finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VPT training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='VPT parser')

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
CUDA_VISIBLE_DEVICES=2,3 torchrun \
        --nproc_per_node=2 \
        train.py \
        CIFAR_VPT \
        --batch-size 64 \
        --tensorboard True
        
CUDA_VISIBLE_DEVICES=2,3 nohup torchrun \
        --nproc_per_node=2 \
        train.py \
        CIFAR_VPT \
        --batch-size 64 \
        --tensorboard True \
        > experiment_1.out \
        &
'''
