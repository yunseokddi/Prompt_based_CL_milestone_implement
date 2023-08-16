def CIFAR100_get_args_parser(subparsers):
    subparsers.add_argument('--batch-size', default=24, type=int, help='Batch size per device')
    subparsers.add_argument('--epochs', default=50, type=int)

    # Model parameters
    subparsers.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                            help='Name of model to train')
    subparsers.add_argument('--input-size', default=224, type=int, help='images input size')
    subparsers.add_argument('--pretrained', default=True, help='Load pretrained model or not')
    subparsers.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    subparsers.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

    # Tensorboard
    subparsers.add_argument('--tensorboard', default=False, type=bool)

    # Data parameters
    subparsers.add_argument('--data-path', default='/home/dorosee/yunseok/data/CL_milestone_dataset', type=str,
                            help='dataset path')
    subparsers.add_argument('--dataset', default='Split-CIFAR100', type=str, help='dataset name')
    subparsers.add_argument('--shuffle', default=False, help='shuffle the data order')
    subparsers.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
    subparsers.add_argument('--checkpoint_dir', default='checkpoint', help='path where to save, empty for no saving')
    subparsers.add_argument('--device', default='cuda', help='device to use for training / testing')
    subparsers.add_argument('--seed', default=42, type=int)
    subparsers.add_argument('--eval', action='store_true', help='Perform evaluation only')
    subparsers.add_argument('--num_workers', default=4, type=int)
    subparsers.add_argument('--pin-mem', action='store_true',
                            help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    subparsers.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                            help='')
    subparsers.set_defaults(pin_mem=True)

    # CL
    subparsers.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    subparsers.add_argument('--upper_bound_flag', default=False, action='store_true', help='Upper bound')
    subparsers.add_argument('--memory', type=int, default=0, help="size of memory for replay")
    subparsers.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    subparsers.add_argument('--DW', default=False, action='store_true', help='dataset balancing')
    subparsers.add_argument('--prompt_param', nargs="+", type=float, default=[1, 1, 1],
                            help="e prompt pool size, e prompt length, g prompt length")

    # distributed training parameters
    subparsers.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
    subparsers.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Continual learning parameters
    subparsers.add_argument('--num_tasks', default=10, type=int, help='number of sequential tasks')
    subparsers.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
    subparsers.add_argument('--task_inc', default=False, type=bool, help='if doing task incremental')