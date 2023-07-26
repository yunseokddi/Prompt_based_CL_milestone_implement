import random

from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset
from .datasets import *
from utils.utils import get_world_size, get_rank

class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes

    def __call__(self, img):
        return self.lambd(img, self.nb_classes)


class ContinualDataLoader(object):
    def __init__(self, args):
        self.args = args
        self.dataloader = []
        self.class_mask = [] if self.args.task_inc or self.args.train_mask else None

        self.transform_train = self.build_transform(True)
        self.transform_val = self.build_transform(False)


        if self.args.dataset.startswith('Split-'):
            self.dataset_train, self.dataset_val = self.get_dataset(self.args.dataset.replace('Split-', ''))

            self.args.nb_classes = len(self.dataset_val.classes)

            splited_dataset, class_mask = self.split_single_dataset()
        else:
            if self.args.dataset == '5-datasets':
                dataset_list = ['SVHN', 'MNIST', 'CIFAR10', 'NotMNIST', 'FashionMNIST']
            else:
                dataset_list = self.args.dataset.split(',')

            if self.args.shuffle:
                random.shuffle(dataset_list)
            print(dataset_list)

            self.args.nb_classes = 0

        for i in range(self.args.num_tasks):
            if self.args.dataset.startswith('Split-'):
                self.dataset_train, self.dataset_val = splited_dataset[i]

            else:
                self.dataset_train, self.dataset_val = self.get_dataset(dataset_list[i])

                transform_target = Lambda(self.target_transform, self.args.nb_classes)

                if self.class_mask is not None:
                    self.class_mask.append([i+self.args.nb_classes for i in range(len(self.dataset_val.classes))])
                    self.args.nb_classes += len(self.dataset_val.classes)

                if not self.args.task_inc:
                    self.dataset_train.target_transform = transform_target
                    self.dataset_val.target_transform = transform_target

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

            self.dataloader.append({'train': self.data_loader_train, 'val': self.data_loader_val})

    def get_dataloader(self):
        return self.dataloader, self.class_mask

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

    def split_single_dataset(self):
        nb_classes = len(self.dataset_val.classes)
        assert nb_classes % self.args.num_tasks == 0
        classes_per_task = nb_classes // self.args.num_tasks

        labels = [i for i in range(nb_classes)]

        split_datasets = list()
        mask = list()

        if self.args.shuffle:
            random.shuffle(labels)

        for _ in range(self.args.num_tasks):
            train_split_indices = []
            test_split_indices = []

            scope = labels[:classes_per_task]
            labels = labels[classes_per_task:]

            mask.append(scope)

            for k in range(len(self.dataset_train.targets)):
                if int(self.dataset_train.targets[k]) in scope:
                    train_split_indices.append(k)

            for h in range(len(self.dataset_val.targets)):
                if int(self.dataset_val.targets[h]) in scope:
                    test_split_indices.append(h)

            subset_train, subset_val = Subset(self.dataset_train, train_split_indices), Subset(self.dataset_val,
                                                                                               test_split_indices)

            split_datasets.append([subset_train, subset_val])

        return split_datasets, mask

    def build_transform(self, is_train):
        resize_image = self.args.input_size > 32

        if is_train:
            scale = (0.05, 1.0)
            ratio = (3. / 4., 4. / 3.)
            transform = transforms.Compose([
                transforms.RandomResizedCrop(self.args.input_size, scale=scale, ratio=ratio),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ])
            return transform

        t = []

        if resize_image:
            size = int((256 / 224) * self.args.input_size)
            t.append(
                transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(self.args.input_size))
        t.append(transforms.ToTensor())

        return transforms.Compose(t)

    def target_transform(self, x, nb_classes):
        return x + nb_classes
