import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import os
import model.model

import warnings

from parse_config import CIFAR100_get_args_parser
from pathlib import Path
from utils.utils import init_distributed_mode
from data_loader.data_loaders import ContinualDataLoader
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
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

    print("Creating model: {}".format(args.model))

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
    )

    original_model.to(device)
    model.to(device)

    if args.freeze:
        for p in original_model.parameters():
            p.requires_grad = False

        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    print(args)

    model_without_ddp = model
    print(args.distributed)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("number of params: {}".format(n_parameters))

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size

    args.lr = args.lr * global_batch_size / 256.0

    optimizer = create_optimizer(args, model_without_ddp)

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None
    else:
        lr_scheduler = None
        assert "Check your learning rate scheduler"

    criterion = torch.nn.CrossEntropyLoss().to(device)

    trainer = Trainer(model, model_without_ddp, original_model, criterion, data_loader, optimizer, lr_scheduler, device
                      , class_mask, args)

    if args.eval:
        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir,
                                           os.path.join(args.checkpoint_dir, 'task{}_checkpoint.pth'.format(task_id + 1)))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.module.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return

            _ = trainer._valid_epoch(data_loader, task_id)

        return

    print("Start training for {} epochs".format(args.epochs))
    start_time = time.time()

    trainer.train()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")


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
'''
CUDA_VISIBLE_DEVICES=2,3 nohup python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --use_env train.py \
        cifar100_l2p \
        --model vit_base_patch16_224 \
        --batch-size 128 \
        > experiment_2.out \
        &
        
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --use_env train.py \
        cifar100_l2p \
        --model vit_base_patch16_224 \
        --batch-size 128
        
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py cifar100_l2p --checkpoint_dir checkpoint --eval
'''
