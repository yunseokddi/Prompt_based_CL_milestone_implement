import numpy as np
import torch


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

            if self.args.prompt_pool and self.args.shared_prompt_key:
                if task_id > 0:
                    prev_start = (task_id - 1) * self.args.top_k
                    prev_end = task_id * self.args.top_k

                    cur_start = prev_end
                    cur_end = (task_id + 1) * self.args.top_k

                    with torch.no_grad():
                        if self.args.distributed:
                            self.model.module.e_prompt.prompt_key.grad.zero_()
                            self.model.module.e_prompt.prompt_key[cur_idx] = self.model.module.e_prompt.prompt_key[prev_idx]
                            self.optimizer.param_groups[0]['params'] = self.model.module.parameters()
                        else:
                            self.model.e_prompt.prompt_key.grad.zero_()
                            self.model.e_prompt.prompt_key[cur_idx] = self.model.e_prompt.prompt_key[prev_idx]
                            self.optimizer.param_groups[0]['params'] = self.model.parameters()