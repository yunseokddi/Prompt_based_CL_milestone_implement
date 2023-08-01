import numpy as np
import torch
import math
import sys
import os

from timm.optim import create_optimizer
from utils.utils import get_world_size, AverageMeter, save_on_master, is_main_process
from tqdm import tqdm
from timm.utils import accuracy
from pathlib import Path


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
                self.optimizer = create_optimizer(self.args, self.model)

            for epoch in range(self.args.epochs):
                self._train_epoch(data_loader=self.data_loader[task_id]['train'], epoch=epoch, task_id=task_id)

                if self.lr_scheduler:
                    self.lr_scheduler.step(epoch)

            self._valid_epoch(self.data_loader, task_id)

            if self.args.output_dir and is_main_process():
                Path(os.path.join(self.args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
                checkpoint_path = os.path.join(self.args.output_dir, self.args.checkpoint_dir,
                                               'task{}_checkpoint.pth'.format(task_id + 1))
                state_dict = {
                    'model': self.model_without_ddp.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'args': self.args,
                }

                if self.args.sched is not None and self.args.sched != 'constant':
                    state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()

                save_on_master(state_dict, checkpoint_path)

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

            loss = self.criterion(logits, target)

            avg_loss.update(loss.data, input.size(0))

            if self.args.pull_constraint and 'reduce_sim' in output:
                loss = loss - self.args.pull_constraint_coeff * output['reduce_sim']

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            avg_acc1.update(acc1, input.size(0))
            avg_acc5.update(acc5, input.size(0))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad)
            self.optimizer.step()

            torch.cuda.synchronize()

            errors = {
                'Task ID': task_id,
                'Epoch': epoch,
                'Train loss': avg_loss.avg.item(),
                'ACC@1': avg_acc1.avg.item(),
                'ACC@5': avg_acc5.avg.item()
            }

            tq_train.set_postfix(errors)

        print("Task ID : {}, Epoch : {}, Train loss : {:.3f}, ACC@1 : {:.3f}, ACC@5 : {:.3f}".format(task_id, epoch,
                                                                                                     avg_loss.avg.item(),
                                                                                                     avg_acc1.avg.item(),
                                                                                                     avg_acc5.avg.item()))

        return True

    @torch.no_grad()
    def _valid_epoch(self, data_loader, task_id):
        for i in range(task_id + 1):
            avg_loss = AverageMeter()
            avg_acc1 = AverageMeter()
            avg_acc5 = AverageMeter()

            tq_val = tqdm(data_loader[i]['val'], total=len(data_loader[i]['val']))
            criterion = torch.nn.CrossEntropyLoss()

            self.model.eval()
            self.original_model.eval()

            with torch.no_grad():
                for input, target in tq_val:
                    input = input.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)

                    if self.original_model is not None:
                        output = self.original_model(input)
                        cls_features = output['pre_logits']

                    else:
                        cls_features = None

                    output = self.model(input, task_id=task_id, cls_features=cls_features)
                    logits = output['logits']

                    if self.args.task_inc and self.class_mask is not None:
                        # adding mask to output logits
                        mask = self.class_mask[task_id]
                        mask = torch.tensor(mask, dtype=torch.int64).to(self.device)
                        logits_mask = torch.ones_like(logits, device=self.device) * float('-inf')
                        logits_mask = logits_mask.index_fill(1, mask, 0.0)
                        logits = logits + logits_mask

                    loss = criterion(logits, target)

                    avg_loss.update(loss.data, input.size(0))

                    acc1, acc5 = accuracy(logits, target, topk=(1, 5))

                    avg_acc1.update(acc1, input.size(0))
                    avg_acc5.update(acc5, input.size(0))

                errors = {
                    'Task ID': task_id,
                    'Val loss': avg_loss.avg.item(),
                    'ACC@1': avg_acc1.avg.item(),
                    'ACC@5': avg_acc5.avg.item()
                }

                tq_val.set_postfix(errors)

            print("Task ID : {}, Val loss : {:.3f}, ACC@1 : {:.3f}, ACC@5 : {:.3f}".format(task_id, avg_loss.avg.item(),
                                                                                           avg_acc1.avg.item(),
                                                                                           avg_acc5.avg.item()))

        return True
