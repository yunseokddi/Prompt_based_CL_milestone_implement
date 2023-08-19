def get_args_parser(subparsers):
    # subparsers.add_argument('--', default=, type=, help='')
    subparsers.add_argument('--batch-size', default=24, type=int, help='Batch size per device')
    subparsers.add_argument('--epochs', default=50, type=int)

    # Model parameters
    subparsers.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                            help='Name of model to train')

    subparsers.add_argument('--DBG', default=False, type=bool, help='Debugging mode')

    # distributed training parameters
    subparsers.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
    subparsers.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Data parameters
    subparsers.add_argument('--dataset', default='CIFAR100', type=str, help='dataset name')
    subparsers.add_argument('--data_path', default='/home/dorosee/yunseok/data/VPT_dataset', type=str, help='dataset path')
    subparsers.add_argument('--output_dir', default='./output', type=str, help='Model result dir')
    subparsers.add_argument('--device', default='cuda', help='device to use for training / testing')
    subparsers.add_argument('--class_num', default=100, type=int)
    subparsers.add_argument('--input_size', default=224, type=int)
    subparsers.add_argument('--seed', default=42, type=int)
    subparsers.add_argument('--gpu', default=[2,3], type=list)
    subparsers.add_argument('--num_workers', default=16, type=int)
    subparsers.add_argument('--pin-mem', action='store_true',
                            help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    subparsers.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                            help='')
    subparsers.set_defaults(pin_mem=True)

    # Tensorboard
    subparsers.add_argument('--tensorboard', default=False, type=bool)