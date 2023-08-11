import sys
import warnings
import argparse
import torch
import torch.backends.cudnn
import random
import numpy as np
import time
import datetime
import os
import model.model

from parse_config import CIFAR100_get_args_parser, imr_get_args_parser
from pathlib import Path
from utils.utils import init_distributed_mode
from data_loader.data_loaders import ContinualDataLoader
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from trainer.trainer import Trainer
from tensorboard_logger import configure

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

    print(f"Creating model: {args.model}")

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
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
    )

    original_model.to(device)
    model.to(device)

    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False

        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    # print(args)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: {}'.format(n_parameters))

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

    trainer = Trainer(model, model_without_ddp, original_model, criterion, data_loader, optimizer, lr_scheduler, device,
                      class_mask, args)

    if args.eval:
        total_acc_1 = []
        total_acc_5 = []

        forgetting = []

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir,
                                           os.path.join(args.checkpoint_dir,
                                                        'task{}_checkpoint.pth'.format(task_id + 1)))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.module.load_state_dict(checkpoint['model'])

            else:
                print('No checkpoint found at:', checkpoint_path)
                return

            acc1_list, acc5_list = trainer._valid_epoch(data_loader, task_id)

            avg_acc1 = sum(acc1_list) / len(acc1_list)
            avg_acc5 = sum(acc5_list) / len(acc5_list)
            print("-" * 80)
            print("Task ID : {}, AVG ACC 1 : {:.3f}, AVG ACC 5 : {:.3f}".format(task_id + 1, avg_acc1, avg_acc5))
            print("-" * 80)
            print("")

            total_acc_1.append(avg_acc1)
            total_acc_5.append(avg_acc5)

            forgetting.append(acc1_list)

        print("Total AVG ACC 1 : {:.3f}".format(sum(total_acc_1) / len(total_acc_1)))
        print("Total AVG ACC 5 : {:.3f}".format(sum(total_acc_5) / len(total_acc_5)))

        num = 0

        forgetting_list = []

        for i in range(args.num_tasks - 1):
            max_num = -1.0
            for j in range(num, args.num_tasks):
                if forgetting[j][i] > max_num:
                    max_num = forgetting[j][i]

            forgetting_list.append(max_num - forgetting[args.num_tasks - 1][i])

            num += 1

        print("Total : forgetting : {:.3f}".format(sum(forgetting_list) / len(forgetting_list)))

        return

    print("Start training for {} epochs".format(args.epochs))
    start_time = time.time()

    trainer.train()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_dualprompt':
        config_parser = subparser.add_parser('cifar100_dualprompt', help='Split-CIFAR100 DualPrompt configs')
        CIFAR100_get_args_parser(config_parser)

    elif config == 'imr_dualprompt':
        config_parser = subparser.add_parser('imr_dualprompt', help='Split-ImageNet-R DualPrompt configs')
        imr_get_args_parser(config_parser)


    else:
        assert "Check dataset"

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.tensorboard:
        configure("runs/%s" % (args.dataset))

    main(args)

    sys.exit(0)

'''

---------------------- Split-CIFAR100 train ----------------------
CUDA_VISIBLE_DEVICES=2,3 nohup python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --use_env train.py \
        cifar100_dualprompt \
        --model vit_base_patch16_224 \
        --batch-size 128 \
        --tensorboard True \
        > experiment_1.out \
        &
        
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --use_env train.py \
        cifar100_dualprompt \
        --model vit_base_patch16_224 \
        --batch-size 128 \
        --tensorboard True
        
---------------------- Split-ImageNet-R train ----------------------
CUDA_VISIBLE_DEVICES=2,3 nohup python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --use_env train.py \
        imr_dualprompt \
        --model vit_base_patch16_224 \
        --batch-size 128 \
        --tensorboard True \
        > imr_experiment_1.out \
        &
        
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --use_env train.py \
        imr_dualprompt \
        --model vit_base_patch16_224 \
        --batch-size 128 \
        --tensorboard True

---------------------- Split-CIFAR100 test ----------------------
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py cifar100_dualprompt --checkpoint_dir checkpoint_cifar100 --eval

---------------------- Split-ImageNet-R test ----------------------
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py imr_dualprompt --checkpoint_dir checkpoint_imr --eval
'''
