from torchvision import datasets, transforms
from .datasets import *
from utils.utils import get_world_size, get_rank
from torch.utils.data.dataset import Subset
from collections import Counter


class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset

        self.transform_train = self.build_transform(True)
        self.transform_val = self.build_transform(False)

        self.dataset_train, self.dataset_val = self.get_dataset(self.dataset)

        if self.args.distributed and get_world_size() > 1:
            num_tasks = get_world_size()
            global_rank = get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                self.dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

            sampler_val = torch.utils.data.SequentialSampler(self.dataset_val)

        else:
            sampler_train = torch.utils.data.RandomSampler(self.dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(self.dataset_val)

        self.data_loader_train = torch.utils.data.DataLoader(
            self.dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        self.data_loader_val = torch.utils.data.DataLoader(
            self.dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        print("Dataset name : {}".format(self.args.dataset))
        print("Num of train data : {}".format(len(self.dataset_train)))
        print("Num of val data : {}".format(len(self.dataset_val)))

    def get_dataloader(self):
        return self.data_loader_train, self.data_loader_val

    def get_dataset(self, dataset):
        if dataset == 'CIFAR100':
            dataset_train = datasets.CIFAR100(self.args.data_path, train=True, download=True,
                                              transform=self.transform_train)
            dataset_val = datasets.CIFAR100(self.args.data_path, train=False, download=True,
                                            transform=self.transform_val)

        elif dataset == 'CIFAR10':
            dataset_train = datasets.CIFAR10(self.args.data_path, train=True, download=True,
                                             transform=self.transform_train)
            dataset_val = datasets.CIFAR10(self.args.data_path, train=False, download=True,
                                           transform=self.transform_val)

        elif dataset == 'MNIST':
            dataset_train = MNIST_RGB(self.args.data_path, train=True, download=True,
                                      transform=self.transform_train)
            dataset_val = MNIST_RGB(self.args.data_path, train=False, download=True,
                                    transform=self.transform_val)

        elif dataset == 'FashionMNIST':
            dataset_train = FashionMNIST(self.args.data_path, train=True, download=True,
                                         transform=self.transform_train)
            dataset_val = FashionMNIST(self.args.data_path, train=False, download=True,
                                       transform=self.transform_val)

        elif dataset == 'SVHN':
            dataset_train = SVHN(self.args.data_path, split='train', download=True,
                                 transform=self.transform_train)
            dataset_val = SVHN(self.args.data_path, split='test', download=True, transform=self.transform_val)

        elif dataset == 'NotMNIST':
            dataset_train = NotMNIST(self.args.data_path, train=True, download=True,
                                     transform=self.transform_train)
            dataset_val = NotMNIST(self.args.data_path, train=False, download=True,
                                   transform=self.transform_val)

        elif dataset == 'Flower102':
            dataset_train = Flowers102(self.args.data_path, split='train', download=True,
                                       transform=self.transform_train)
            dataset_val = Flowers102(self.args.data_path, split='test', download=True,
                                     transform=self.transform_val)

        elif dataset == 'Cars196':
            dataset_train = StanfordCars(self.args.data_path, split='train', download=True,
                                         transform=self.transform_train)
            dataset_val = StanfordCars(self.args.data_path, split='test', download=True,
                                       transform=self.transform_val)

        elif dataset == 'CUB200':
            dataset_train = CUB200(self.args.data_path, train=True, download=True,
                                   transform=self.transform_train).data
            dataset_val = CUB200(self.args.data_path, train=False, download=True,
                                 transform=self.transform_val).data

        elif dataset == 'Scene67':
            dataset_train = Scene67(self.args.data_path, train=True, download=True,
                                    transform=self.transform_train).data
            dataset_val = Scene67(self.args.data_path, train=False, download=True,
                                  transform=self.transform_val).data

        elif dataset == 'TinyImagenet':
            dataset_train = TinyImagenet(self.args.data_path, train=True, download=True,
                                         transform=self.transform_train).data
            dataset_val = TinyImagenet(self.args.data_path, train=False, download=True,
                                       transform=self.transform_val).data

        elif dataset == 'Imagenet-R':
            dataset_train = Imagenet_R(self.args.data_path, train=True, download=True,
                                       transform=self.transform_train).data
            dataset_val = Imagenet_R(self.args.data_path, train=False, download=True,
                                     transform=self.transform_val).data

        else:
            raise ValueError('Dataset {} not found.'.format(dataset))

        return dataset_train, dataset_val

    def build_transform(self, is_train):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if self.args.input_size == 448:
            resize_dim = 512
            crop_dim = 448
        elif self.args.input_size == 224:
            resize_dim = 256
            crop_dim = 224
        elif self.args.input_size == 384:
            resize_dim = 438
            crop_dim = 384

        if is_train:
            transform = transforms.Compose(
                [
                    transforms.Resize(resize_dim),
                    transforms.RandomCrop(crop_dim),
                    transforms.RandomHorizontalFlip(0.5),
                    # tv.transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                    # tv.transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(resize_dim),
                    transforms.CenterCrop(crop_dim),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

        return transform

    def get_class_num(self):
        return self.args.class_num

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""

        cls_num = self.get_class_num()

        return [1.0] * cls_num