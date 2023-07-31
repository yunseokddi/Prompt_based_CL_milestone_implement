import numpy as np
import torch

from timm.optim import create_optimizer
from utils.utils import get_world_size, AverageMeter
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, model_without_ddp, original_model, criterion, data_loader, optimizer, lr_scheduler,
                 device,
                 class_mask=None, args=None):
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

        self.acc_matrix = np.zeros((self.args.num_tasks, self.args.num_tasks))

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
                        cur_idx = (
                            slice(None), slice(None),
                            slice(cur_start, cur_end)) if self.args.use_prefix_tune_for_e_prompt else (
                            slice(None), slice(cur_start, cur_end))
                        prev_idx = (slice(None), slice(None),
                                    slice(prev_start, prev_end)) if self.args.use_prefix_tune_for_e_prompt else (
                            slice(None), slice(prev_start, prev_end))

                        with torch.no_grad():
                            if self.args.distributed:
                                # self.model.module.e_prompt.prompt.grad.zero_()
                                self.model.module.e_prompt.prompt[cur_idx] = self.model.module.e_prompt.prompt[prev_idx]
                                self.optimizer.param_groups[0]['params'] = self.model.module.parameters()
                            else:
                                # self.model.e_prompt.prompt.grad.zero_()
                                self.model.e_prompt.prompt[cur_idx] = self.model.e_prompt.prompt[prev_idx]
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
                            self.model.module.e_prompt.prompt_key.grad.zero_()
                            self.model.module.e_prompt.prompt_key[cur_idx] = self.model.module.e_prompt.prompt_key[
                                prev_idx]
                            self.optimizer.param_groups[0]['params'] = self.model.module.parameters()
                        else:
                            self.model.e_prompt.prompt_key.grad.zero_()
                            self.model.e_prompt.prompt_key[cur_idx] = self.model.e_prompt.prompt_key[prev_idx]
                            self.optimizer.param_groups[0]['params'] = self.model.parameters()

            # Create new optimizer for each task to clear optimizer status
            if task_id > 0 and self.args.reinit_optimizer:
                optimizer = create_optimizer(self.args, self.model)

            for epoch in range(self.args.epochs):
                self._train_epoch(data_loader=self.data_loader[task_id]['train'], epoch=epoch, task_id=task_id)

    def _train_epoch(self, data_loader, epoch, task_id):
        avg_loss = AverageMeter()
        avg_acc1 = AverageMeter()
        avg_acc5 = AverageMeter()

        tq_train = tqdm(data_loader, total=len(data_loader))
        self.model.train(True)
        self.original_model.eval()

        if self.args.distributed and get_world_size() > 1:
            data_loader.sampler.set_epoch(epoch)

        for input, target in tq_train:
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    cls_features = output['pre_logits']

                else:
                    cls_features = None

            output = self.model(input, task_id=task_id, cls_features=cls_features, train=True)
            logits = output['logits']

            if self.args.train_mask and self.class_mask is not None:
                mask = self.class_mask[task_id]
                not_mask = np.setdiff1d(np.arange(self.args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                print("Hello")