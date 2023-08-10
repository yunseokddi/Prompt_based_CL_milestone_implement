import torch

from timm.models import create_model

model = 'vit_base_patch16_224'
checkpoint = "./checkpoint/best_checkpoint.pth"

checkpoint_model = torch.load(checkpoint)

original_model = create_model(
        model,
        pretrained=True,
        num_classes = 1000
    )

print(f"Creating model: {model}")