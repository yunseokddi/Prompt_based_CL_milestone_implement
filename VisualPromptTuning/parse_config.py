def get_args_parser(subparsers):
    # subparsers.add_argument('--', default=, type=, help='')
    subparsers.add_argument('--batch-size', default=128, type=int, help='Batch size per device')
    subparsers.add_argument('--epochs', default=50, type=int)
    subparsers.add_argument('--load_pretrain', default=True, type=bool, help='Load pretrained model')
    subparsers.add_argument('--vis', default=False, type=bool, help='Visualization')

    # Model parameters
    subparsers.add_argument('--model', default='sup_vitb16_imagenet21k', type=str,
                            help='Name of model to train')
    subparsers.add_argument('--model_type', default='vit', type=str,
                            help='Name of model type')
    subparsers.add_argument('--DBG', default=False, type=bool, help='Debugging mode')
    subparsers.add_argument('--TRANSFER_TYPE', default="prompt", type=str, help='Transfer learning type')
    subparsers.add_argument('--MLP_NUM', default=0, type=int,
                            help='number of MLP')

    # distributed training parameters
    subparsers.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
    subparsers.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Data parameters
    subparsers.add_argument('--dataset', default='CIFAR100', type=str, help='dataset name')
    subparsers.add_argument('--data_path', default='/home/dorosee/yunseok/data/VPT_dataset', type=str,
                            help='dataset path')
    subparsers.add_argument('--output_dir', default='./output', type=str, help='Model result dir')
    subparsers.add_argument('--weight_dir',
                            default='/home/dorosee/yunseok/clone_src/CL_milestone_implement/VisualPromptTuning/checkpoint/',
                            type=str, help='Model result dir')
    subparsers.add_argument('--resume_dir',
                            default='',
                            type=str, help='Model result dir')
    subparsers.add_argument('--device', default='cuda', help='device to use for training / testing')
    subparsers.add_argument('--class_num', default=100, type=int)
    subparsers.add_argument('--input_size', default=224, type=int)
    subparsers.add_argument('--seed', default=42, type=int)
    subparsers.add_argument('--gpu', default=[2, 3], type=list)
    subparsers.add_argument('--num_workers', default=16, type=int)
    subparsers.add_argument('--pin-mem', action='store_true',
                            help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    subparsers.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                            help='')
    subparsers.add_argument('--CLASS_WEIGHTS_TYPE',
                            default='none',
                            type=str, help='cls weight')
    subparsers.set_defaults(pin_mem=True)

    # Tensorboard
    subparsers.add_argument('--tensorboard', default=False, type=bool)

    # Prompt parameters
    subparsers.add_argument('--NUM_TOKENS', default=50, type=int, help='Num of prompt tokens')
    subparsers.add_argument('--LOCATION', default="prepend", type=str, help='Num of prompt tokens')
    # prompt initalizatioin:
    # (1) default "random"
    # (2) "final-cls" use aggregated final [cls] embeddings from training dataset
    # (3) "cls-nolastl": use first 12 cls embeddings (exclude the final output) for deep prompt
    # (4) "cls-nofirstl": use last 12 cls embeddings (exclude the input to first layer)

    subparsers.add_argument('--INITIATION', default="random", type=str, help='Num of prompt tokens')
    subparsers.add_argument('--CLSEMB_FOLDER', default="", type=str, help='Num of prompt tokens')
    subparsers.add_argument('--CLSEMB_PATH', default="", type=str, help='Num of prompt tokens')
    subparsers.add_argument('--PROJECT', default=-1, type=int, help='Projection mlp hidden dim')
    subparsers.add_argument('--DEEP', default=False, type=bool,
                            help='Whether do deep prompt or not, only for prepend location')

    subparsers.add_argument('--NUM_DEEP_LAYERS', default=None, type=int,
                            help='if set to be an int, then do partial-deep prompt tuning')
    subparsers.add_argument('--REVERSE_DEEP', default=False, type=bool,
                            help='if to only update last n layers, not the input layer')
    subparsers.add_argument('--DEEP_SHARED', default=False, type=bool,
                            help='if true, all deep layers will be use the same prompt emb')
    subparsers.add_argument('--FORWARD_DEEP_NOEXPAND', default=False, type=bool,
                            help='if true, will not expand input sequence for layers without prompt')

    # how to get the output emb for cls head:
    # original: follow the orignial backbone choice,
    # img_pool: image patch pool only
    # prompt_pool: prompt embd pool only
    # imgprompt_pool: pool everything but the cls token

    subparsers.add_argument('--VIT_POOL_TYPE', default="original", type=str,
                            help='Prompt pool type')
    subparsers.add_argument('--DROPOUT', default=0.0, type=float,
                            help='Prompt training dropout rate')
    subparsers.add_argument('--SAVE_FOR_EACH_EPOCH', default=False, type=bool,
                            help='if true, it save prompt weight for each epoch')

    # Optimizer parameters // Fix help comment
    subparsers.add_argument('--LOSS', default="softmax", type=str,
                            help='Prompt pool type')
    subparsers.add_argument('--LOSS_ALPHA', default=0.01, type=float,
                            help='Prompt pool type')
    subparsers.add_argument('--OPTIMIZER', default="sgd", type=str,
                            help='Prompt pool type')  # or "adamw"
    subparsers.add_argument('--MOMENTUM', default=0.9, type=float,
                            help='Prompt pool type')
    subparsers.add_argument('--WEIGHT_DECAY', default=0.0001, type=float,
                            help='Prompt pool type')
    subparsers.add_argument('--WEIGHT_DECAY_BIAS', default=0, type=int,
                            help='Prompt pool type')
    subparsers.add_argument('--PATIENCE', default=300, type=int,
                            help='Prompt pool type')
    subparsers.add_argument('--SCHEDULER', default="cosine", type=str,
                            help='Prompt pool type')
    subparsers.add_argument('--BASE_LR', default=0.01, type=float,
                            help='Prompt pool type')
    subparsers.add_argument('--BIAS_MULTIPLIER', default=1., type=float,
                            help='Prompt pool type')  # for prompt + bias
    subparsers.add_argument('--WARMUP_EPOCH', default=5, type=int,
                            help='Prompt pool type')
    subparsers.add_argument('--TOTAL_EPOCH', default=30, type=int,
                            help='Prompt pool type')
    subparsers.add_argument('--LOG_EVERY_N', default=1000, type=int,
                            help='Prompt pool type')
    subparsers.add_argument('--DBG_TRAINABLE', default=False, type=bool,
                            help='if true, it save prompt weight for each epoch')  # if True, will print the name of trainable params
