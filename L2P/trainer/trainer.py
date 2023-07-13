import numpy as np
import torch
import math
import os

from timm.optim import create_optimizer
from utils.utils import get_world_size, MetricLogger, SmoothedValue

class Trainer(object):
    def __init__(self, model, model_without_ddp, original_model,
                 criterion, data_loader, optimizer, lr_scheduler, device,
                 class_mask=None, args=None, ):
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.original_model = original_model
        self.criterion = criterion
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.class_mask = class_mask
        self.args = args

        self.acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    def train(self):
        for task_id in range(self.args.num_tasks):

            # Transfer previous learned prompt params to the new prompt
            if self.args.prompt_pool and self.args.shared_prompt_pool:
                if task_id > 0:
                    prev_start = (task_id - 1) * self.args.top_k
                    prev_end = task_id * self.args.top_k

                    cur_start = prev_end
                    cur_end = (task_id + 1) * self.args.top_k

                    if (prev_end > self.args.size) or (cur_end > self.args.size):
                        pass

                    else:
                        cur_idx = (slice(cur_start, cur_end))
                        prev_idx = (slice(prev_start, prev_end))

                        with torch.no_grad():
                            if self.args.distributed:
                                self.model.module.prompt.prompt.grad.zero_()
                                self.model.module.prompt.prompt[cur_idx] = self.model.module.prompt.prompt[prev_idx]
                                self.optimizer.param_groups[0]['params'] = self.model.module.parameters()

                            else:
                                self.model.prompt.prompt.grad.zero_()
                                self.model.prompt.prompt[cur_idx] = self.model.prompt.prompt[prev_idx]
                                self.optimizer.param_groups[0]['params'] = self.model.parameters()

            # Transfer previous learned prompt param keys to the new prompt
            if self.args.prompt_pool and self.args.shared_prompt_key:
                if task_id > 0:
                    prev_start = (task_id - 1) * self.args.top_k
                    prev_end = task_id * self.args.top_k

                    cur_start = prev_end
                    cur_end = (task_id + 1) * self.args.top_k

                    with torch.no_grad():
                        if self.args.distributed:
                            self.model.module.prompt.prompt_key.grad.zero_()
                            self.model.module.prompt.prompt_key[cur_idx] = self.model.module.prompt.prompt_key[prev_idx]
                            self.optimizer.param_groups[0]['params'] = self.model.module.parameters()

                        else:
                            self.model.prompt.prompt_key.grad.zero_()
                            self.model.prompt.prompt_key[cur_idx] = self.model.prompt.prompt_key[prev_idx]
                            self.optimizer.param_groups[0]['params'] = self.model.parameters()

            # Create new optimizer for each task to clear optimizer status
            if task_id > 0 and self.args.reinit_optimizer:
                self.optimizer = create_optimizer(self.args, self.model)

            for epoch in range(self.args.epochs):
                self._train_epoch(data_loader=self.data_loader[task_id]['train'], epoch=epoch)

    def _train_epoch(self, data_loader, epoch):
        self.model.train(True)
        self.original_model.eval()

        if self.args.distributed and get_world_size() > 1:
            data_loader.sampler.set_epoch(epoch)

        # metric_logger = MetricLogger(delimiter="  ")
        # metric_logger.add_meter('Lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        # metric_logger.add_meter('Loss', SmoothedValue(window_size=1, fmt='{value:.4f}'))
        # header = f'Train: Epoch[{epoch + 1:{int(math.log10(self.args.epochs)) + 1}}/{self.args.epochs}]'
        #
        # for input, target in metric_logger.log_every(data_loader, self.args.print_freq, header):
        #     input = input.to(self.device, non_blocking=True)
        #     target = target.to(self.device, non_blocking=True)
        #
        #     with torch.no_grad():
        #         if self.original_model is not None:
        #             output = self.original_model(input)
        #             cls_features = output['pre_logits']
        #
        #         else:
        #             cls_features = None

        for input, target in data_loader:
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    cls_features = output['pre_logits']

                else:
                    cls_features = None