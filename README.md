# CL_milestone_implement

This is Pytorch implementation of the Prompt-based Continual Learning with [pytorch-template](https://github.com/victoresque/pytorch-template) in the following milestone papers:

- [L2P] A Baseline for Detecting Missclassified and Out-of-Distribution Examples in Neural Networks | **[CVPR2022]** [[paper]](https://arxiv.org/abs/2112.08654.pdf)

## Installation & requirement

The current version of the code has been tested with `python 3.8.0` on an Ubuntu 18.04 OS with the following versions of Pytorch, Torchvision and timm:

- `pytorch 1.12.0`
- `torchvision 0.13.0`
- `timm 0.6.7`

Further Python-packages used are listed in `requirements.txt`.
Assuming Python and pip are set up, these packages can be installed using:
```bash
pip install -r requirements.txt
```

## Folder Structure
```angular2html
OOD_milestone_implement/
│
├── L2P/ - L2P implementation proj
│   └── data_loader/ - CIFAR100 and 5-dataset loader
│   └── model/ - backbone (ViT) + prompt src
│   └── output/ - checkpoint dir
│   └── trainer/ - Train core src
│   └── utils/ 
│   └── parse_config.py - parser
│   └── train.py - main src
├── ViT/ - ViT implementation proj
├── license
├── requirements.txt
└── train.py - **main script to start training**
```

## Running custom experiments
### 1. L2P
**How to run?**

(1) If you want to full-train
```train
CUDA_VISIBLE_DEVICES=[GPU idx] python -m torch.distributed.launch \
        --nproc_per_node=[Node num] \
        --use_env train.py \
        cifar100_l2p \
        --model vit_base_patch16_224 \
        --batch-size [Batch size] \
```
(2) If you want to evaluate OOD score
```eval
CUDA_VISIBLE_DEVICES=[GPU idx] python -m torch.distributed.launch --nproc_per_node=[Node num] --use_env train.py cifar100_l2p --eval
```

**Result**

(1) CIFAR100
```result
Loading checkpoint from: output/checkpoint/task1_checkpoint.pth
100%|████████████████████████████████████████████████████████████████| 63/63 [00:08<00:00,  7.77it/s]
Task ID : 0, Val loss : 0.260, ACC@1 : 97.400, ACC@5 : 99.900

Loading checkpoint from: output/checkpoint/task2_checkpoint.pth
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.39it/s]
Task ID : 0, Val loss : 0.311, ACC@1 : 94.600, ACC@5 : 99.900
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.36it/s]
Task ID : 1, Val loss : 0.401, ACC@1 : 94.400, ACC@5 : 99.400

Loading checkpoint from: output/checkpoint/task3_checkpoint.pth
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.31it/s]
Task ID : 0, Val loss : 0.369, ACC@1 : 90.900, ACC@5 : 99.800
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.25it/s]
Task ID : 1, Val loss : 0.440, ACC@1 : 92.500, ACC@5 : 98.600
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.18it/s]
Task ID : 2, Val loss : 0.356, ACC@1 : 91.700, ACC@5 : 99.500

Loading checkpoint from: output/checkpoint/task4_checkpoint.pth
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.28it/s]
Task ID : 0, Val loss : 0.380, ACC@1 : 90.900, ACC@5 : 99.800
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.19it/s]
Task ID : 1, Val loss : 0.510, ACC@1 : 88.200, ACC@5 : 98.500
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.15it/s]
Task ID : 2, Val loss : 0.381, ACC@1 : 91.100, ACC@5 : 99.300
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.20it/s]
Task ID : 3, Val loss : 0.387, ACC@1 : 92.000, ACC@5 : 98.400

Loading checkpoint from: output/checkpoint/task5_checkpoint.pth
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.22it/s]
Task ID : 0, Val loss : 0.465, ACC@1 : 88.500, ACC@5 : 99.600
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.18it/s]
Task ID : 1, Val loss : 0.576, ACC@1 : 84.500, ACC@5 : 97.400
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.24it/s]
Task ID : 2, Val loss : 0.471, ACC@1 : 87.300, ACC@5 : 98.400
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.10it/s]
Task ID : 3, Val loss : 0.452, ACC@1 : 89.800, ACC@5 : 97.800
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.22it/s]
Task ID : 4, Val loss : 0.358, ACC@1 : 92.500, ACC@5 : 98.800

Loading checkpoint from: output/checkpoint/task6_checkpoint.pth
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.24it/s]
Task ID : 0, Val loss : 0.499, ACC@1 : 87.500, ACC@5 : 98.900
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.21it/s]
Task ID : 1, Val loss : 0.588, ACC@1 : 84.400, ACC@5 : 97.300
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.12it/s]
Task ID : 2, Val loss : 0.519, ACC@1 : 86.800, ACC@5 : 98.600
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.20it/s]
Task ID : 3, Val loss : 0.513, ACC@1 : 88.600, ACC@5 : 97.400
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.14it/s]
Task ID : 4, Val loss : 0.399, ACC@1 : 91.900, ACC@5 : 98.100
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.17it/s]
Task ID : 5, Val loss : 0.493, ACC@1 : 84.300, ACC@5 : 99.100

Loading checkpoint from: output/checkpoint/task7_checkpoint.pth
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.20it/s]
Task ID : 0, Val loss : 0.510, ACC@1 : 86.600, ACC@5 : 98.300
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.17it/s]
Task ID : 1, Val loss : 0.606, ACC@1 : 84.300, ACC@5 : 97.400
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.22it/s]
Task ID : 2, Val loss : 0.558, ACC@1 : 85.300, ACC@5 : 97.400
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.17it/s]
Task ID : 3, Val loss : 0.556, ACC@1 : 86.800, ACC@5 : 96.600
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.15it/s]
Task ID : 4, Val loss : 0.422, ACC@1 : 90.300, ACC@5 : 97.900
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.18it/s]
Task ID : 5, Val loss : 0.521, ACC@1 : 84.300, ACC@5 : 98.700
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.20it/s]
Task ID : 6, Val loss : 0.490, ACC@1 : 86.800, ACC@5 : 97.900

Loading checkpoint from: output/checkpoint/task8_checkpoint.pth
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.15it/s]
Task ID : 0, Val loss : 0.529, ACC@1 : 85.800, ACC@5 : 98.100
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.12it/s]
Task ID : 1, Val loss : 0.654, ACC@1 : 82.800, ACC@5 : 97.200
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.11it/s]
Task ID : 2, Val loss : 0.574, ACC@1 : 84.500, ACC@5 : 97.900
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.12it/s]
Task ID : 3, Val loss : 0.569, ACC@1 : 86.600, ACC@5 : 96.500
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.13it/s]
Task ID : 4, Val loss : 0.420, ACC@1 : 90.100, ACC@5 : 98.100
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.10it/s]
Task ID : 5, Val loss : 0.534, ACC@1 : 83.600, ACC@5 : 98.500
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.15it/s]
Task ID : 6, Val loss : 0.567, ACC@1 : 84.300, ACC@5 : 97.500
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.10it/s]
Task ID : 7, Val loss : 0.624, ACC@1 : 83.800, ACC@5 : 96.800

Loading checkpoint from: output/checkpoint/task9_checkpoint.pth
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.13it/s]
Task ID : 0, Val loss : 0.584, ACC@1 : 85.400, ACC@5 : 98.100
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.17it/s]
Task ID : 1, Val loss : 0.691, ACC@1 : 82.100, ACC@5 : 96.800
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.10it/s]
Task ID : 2, Val loss : 0.580, ACC@1 : 84.100, ACC@5 : 97.300
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.17it/s]
Task ID : 3, Val loss : 0.628, ACC@1 : 85.300, ACC@5 : 96.200
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.05it/s]
Task ID : 4, Val loss : 0.496, ACC@1 : 88.900, ACC@5 : 97.100
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.16it/s]
Task ID : 5, Val loss : 0.568, ACC@1 : 83.200, ACC@5 : 98.300
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.10it/s]
Task ID : 6, Val loss : 0.647, ACC@1 : 82.900, ACC@5 : 96.700
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.15it/s]
Task ID : 7, Val loss : 0.682, ACC@1 : 83.400, ACC@5 : 96.200
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.06it/s]
Task ID : 8, Val loss : 0.362, ACC@1 : 89.100, ACC@5 : 98.800

Loading checkpoint from: output/checkpoint/task10_checkpoint.pth
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.11it/s]
Task ID : 0, Val loss : 0.599, ACC@1 : 84.400, ACC@5 : 97.900
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.06it/s]
Task ID : 1, Val loss : 0.689, ACC@1 : 82.100, ACC@5 : 96.500
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.10it/s]
Task ID : 2, Val loss : 0.580, ACC@1 : 83.500, ACC@5 : 97.600
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.11it/s]
Task ID : 3, Val loss : 0.625, ACC@1 : 83.800, ACC@5 : 96.600
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.08it/s]
Task ID : 4, Val loss : 0.525, ACC@1 : 86.100, ACC@5 : 97.700
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.04it/s]
Task ID : 5, Val loss : 0.557, ACC@1 : 82.700, ACC@5 : 98.700
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.09it/s]
Task ID : 6, Val loss : 0.656, ACC@1 : 83.000, ACC@5 : 96.000
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.12it/s]
Task ID : 7, Val loss : 0.679, ACC@1 : 82.600, ACC@5 : 96.400
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.14it/s]
Task ID : 8, Val loss : 0.432, ACC@1 : 86.800, ACC@5 : 98.800
100%|████████████████████████████████████████████████████████████████| 63/63 [00:06<00:00, 10.08it/s]
Task ID : 9, Val loss : 0.424, ACC@1 : 87.800, ACC@5 : 98.800
 ```


## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/yunseokddi/CL_milestone_implement/blob/main/LICENSE) for more details

## Reference

- **Project structure**: [https://github.com/jfc43/informative-outlier-mining](https://github.com/jfc43/informative-outlier-mining)
- **L2P**
  - [Original code] [https://github.com/google-research/l2p](https://github.com/google-research/l2p)
    
  - [Pytorch version] [https://github.com/JH-LEE-KR/l2p-pytorch](https://github.com/JH-LEE-KR/l2p-pytorch) 