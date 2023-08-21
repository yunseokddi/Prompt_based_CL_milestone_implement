import os
import numpy as np
import math
import torch
import torch.nn as nn

from .backbone.ViT import VisionTransformer, CONFIGS
from .prompt.ViT import PromptedTransformer
from .mlp import MLP

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
        model.load_from(np.load(os.path.join(args.weight_dir, MODEL_ZOO[args.model])))

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

        self.setup_size()
        self.setup_head(args)

    def setup_size(self):
        self.side = None

    def setup_head(self, args):
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.args.MLP_NUM + [args.class_num],
            special_bias=True
        )

    def build_backbone(self, args, load_pretrain, vis):
        transfer_type = args.TRANSFER_TYPE
        self.enc, self.feat_dim = build_vit_sup_models(
            self.args, load_pretrain, vis
        )
        # linear, prompt, cls, cls+prompt, partial_1
        if transfer_type == "partial-1":
            total_layer = len(self.enc.transformer.encoder.layer)
            # tuned_params = [
            #     "transformer.encoder.layer.{}".format(i-1) for i in range(total_layer)]
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(
                        total_layer - 1) not in k and "transformer.encoder.encoder_norm" not in k:  # noqa
                    p.requires_grad = False
        elif transfer_type == "partial-2":
            total_layer = len(self.enc.transformer.encoder.layer)
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(
                        total_layer - 1) not in k and "transformer.encoder.layer.{}".format(
                    total_layer - 2) not in k and "transformer.encoder.encoder_norm" not in k:  # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            total_layer = len(self.enc.transformer.encoder.layer)
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(
                        total_layer - 1) not in k and "transformer.encoder.layer.{}".format(
                    total_layer - 2) not in k and "transformer.encoder.layer.{}".format(
                    total_layer - 3) not in k and "transformer.encoder.layer.{}".format(
                    total_layer - 4) not in k and "transformer.encoder.encoder_norm" not in k:  # noqa
                    p.requires_grad = False

        elif transfer_type == "linear" or transfer_type == "side":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt" and args.LOCATION == "below":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "embeddings.patch_embeddings.weight" not in k and "embeddings.patch_embeddings.bias" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt+bias":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt-noupdate":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "cls":
            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False

        elif transfer_type == "cls-reinit":
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )

            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False

        elif transfer_type == "cls+prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "cls_token" not in k:
                    p.requires_grad = False

        elif transfer_type == "cls-reinit+prompt":
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "cls_token" not in k:
                    p.requires_grad = False

        # adapter
        elif transfer_type == "adapter":
            for k, p in self.enc.named_parameters():
                if "adapter" not in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            print("Enable all parameters update during training")

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))

    def forward(self, x, return_feature=False):
        if self.froze_enc and self.enc.training:
            self.enc.eval()

        x = self.enc(x)

        if return_feature:
            return x, x
        x = self.head(x)

        return x

    def forward_cls_layerwise(self, x):
        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim
        return x


class PromptedVisionTransformer(VisionTransformer):
    def __init__(self, args, vis):
        assert args.VIT_POOL_TYPE == "original"
        super(PromptedVisionTransformer, self).__init__(
            args, vis)

        self.args = args
        vit_cfg = CONFIGS[self.args.model]
        self.transformer = PromptedTransformer(
            self.args, vit_cfg, vis)
