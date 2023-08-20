import os
import numpy as np
import math
import torch
import torch.nn as nn

from .backbone.ViT import VisionTransformer, CONFIGS
from .prompt.ViT import PromptedTransformer

MODEL_ZOO = {
    "swint_imagenet": "swin_tiny_patch4_window7_224.pth",
    "swint_imagenet_ssl": "moby_swin_t_300ep_pretrained.pth",
    "swins_imagenet": "swin_small_patch4_window7_224.pth",
    "swinb_imagenet_224": "swin_base_patch4_window7_224.pth",
    "swinb_imagenet_384": "swin_base_patch4_window12_384.pth",
    "swinb_imagenet22k_224": "swin_base_patch4_window7_224_22k.pth",
    "swinb_imagenet22k_384": "swin_base_patch4_window12_384_22k.pth",
    "swinl_imagenet22k_224": "swin_large_patch4_window7_224_22k.pth",
    "sup_vitb8": "ViT-B_8.npz",
    "sup_vitb16_224": "ViT-B_16-224.npz",
    "sup_vitb16": "ViT-B_16.npz",
    "sup_vitl16_224": "ViT-L_16-224.npz",
    "sup_vitl16": "ViT-L_16.npz",
    "sup_vitb8_imagenet21k": "imagenet21k_ViT-B_8.npz",
    "sup_vitb32_imagenet21k": "imagenet21k_ViT-B_32.npz",
    "sup_vitb16_imagenet21k": "imagenet21k_ViT-B_16.npz",
    "sup_vitl16_imagenet21k": "imagenet21k_ViT-L_16.npz",
    "sup_vitl32_imagenet21k": "imagenet21k_ViT-L_32.npz",
    "sup_vith14_imagenet21k": "imagenet21k_ViT-H_14.npz",
    "mae_vith14": "mae_pretrain_vit_huge.pth",
    "mae_vitb16": "mae_pretrain_vit_base.pth",
    "mae_vitl16": "mae_pretrain_vit_large.pth",
}


def build_model(args):
    """
    build model here
    """

    train_type = args.model_type

    print("Model type : {}".format(train_type))

    if train_type == "vit":
        model = ViT(args, args.load_pretrain, args.vis)

    else:
        model = None

        assert "Check model type"

    model = model.cuda(device=args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        module=model, device_ids=[args.gpu], output_device=args.gpu,
        find_unused_parameters=True
    )

    return model


def build_vit_sup_models(args, load_pretrain=True, vis=False):
    # image size is the size of actual image
    m2featdim = {
        "sup_vitb16_224": 768,
        "sup_vitb16": 768,
        "sup_vitl16_224": 1024,
        "sup_vitl16": 1024,
        "sup_vitb8_imagenet21k": 768,
        "sup_vitb16_imagenet21k": 768,
        "sup_vitb32_imagenet21k": 768,
        "sup_vitl16_imagenet21k": 1024,
        "sup_vitl32_imagenet21k": 1024,
        "sup_vith14_imagenet21k": 1280,
    }

    model = PromptedVisionTransformer(
        args, vis=vis
    )

    if load_pretrain:
        model.load_from(np.load(os.path.join(args.weight_dir, MODEL_ZOO[args.model_type])))

    return model, m2featdim[args.model]


class ViT(nn.Module):
    def __init__(self, args, load_pretrain=True, vis=False):
        super(ViT, self).__init__()

        self.args = args

        if args.TRANSFER_TYPE == "prompt":
            self.num_token = args.NUM_TOKENS
            self.initiation = args.INITIATION

        self.froze_enc = False

        self.build_backbone(
            self.args, load_pretrain, vis
        )

    def build_backbone(self, args, load_pretrain, vis):
        transfer_type = args.TRANSFER_TYPE
        self.enc, self.feat_dim = build_vit_sup_models(
            self.args, load_pretrain, vis
        )


class PromptedVisionTransformer(VisionTransformer):
    def __init__(self, args, vis):
        assert args.VIT_POOL_TYPE == "original"
        super(PromptedVisionTransformer, self).__init__(
            args, vis)

        self.args = args
        vit_cfg = CONFIGS[self.args.model]
        self.transformer = PromptedTransformer(
            self.args, vit_cfg, vis)
