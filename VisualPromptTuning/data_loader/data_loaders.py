from torchvision import datasets, transforms


class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self.data_dir = args.data_dir

    def build_transform(self, is_train, size):
        normalize = transforms.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if size == 448:
            resize_dim = 512
            crop_dim = 448
        elif size == 224:
            resize_dim = 256
            crop_dim = 224
        elif size == 384:
            resize_dim = 438
            crop_dim = 384

        else:
            assert "Check image size"

        if is_train == "train":
            transform = transforms.transforms.Compose(
                [
                    transforms.transforms.Resize(resize_dim),
                    transforms.transforms.RandomCrop(crop_dim),
                    transforms.transforms.RandomHorizontalFlip(0.5),
                    # tv.transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                    # tv.transforms.RandomHorizontalFlip(),
                    transforms.transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.transforms.Compose(
                [
                    transforms.transforms.Resize(resize_dim),
                    transforms.transforms.CenterCrop(crop_dim),
                    transforms.transforms.ToTensor(),
                    normalize,
                ]
            )
        return transform