import torch
import os

from fvcore.common.checkpoint import Checkpointer
from utils.utils import AverageMeter, get_world_size
from tqdm import tqdm
from timm.utils import accuracy
from tensorboard_logger import log_value


class Trainer(object):
    def __init__(self, model, criterion, data_loader, optimizer, lr_scheduler, device, args):
        self.model = model
        self.criterion = criterion
        self.data_loader = data_loader
        self.train_loader, self.val_loader = data_loader.get_dataloader()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.args = args

        self.cpu_device = torch.device("cpu")

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=self.args.output_dir,
            save_to_disk=True
        )

        if len(args.resume_dir) > 0:
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.weight"]]
            self.checkpointer.load(args.resume_dir, checkpointables)

    def train(self):
        # save the model prompt if required before training
        self.model.eval()
        self.save_prompt(0)

        self.cls_weights = self.data_loader.get_class_weights(
            self.args.CLASS_WEIGHTS_TYPE)

        patience = 0  # if > self.PATIENCE, stop training

        for epoch in range(self.args.TOTAL_EPOCH):
            self._train_epoch(self.train_loader, epoch)

    def _train_epoch(self, train_loder, epoch):
        avg_loss = AverageMeter()
        avg_acc1 = AverageMeter()
        avg_acc5 = AverageMeter()

        tq_train = tqdm(train_loder, total=len(train_loder))
        self.model.train(True)

        if self.args.distributed and get_world_size() > 1:
            train_loder.sampler.set_epoch(epoch)

        for input, target in tq_train:
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.set_grad_enabled(True):
                outputs = self.model(input)

                if self.criterion.is_local():
                    self.model.eval()
                    loss = self.criterion(outputs, target, self.cls_weights, self.model, input)

                else:
                    loss = self.criterion(outputs, target, self.cls_weights)

                if loss == float('inf'):
                    return -1, -1

                elif torch.isnan(loss).any():
                    return -1, -1

            avg_loss.update(loss.data, input.size(0))

            acc1, acc5 = accuracy(outputs, target, topk=(1, 5))

            avg_acc1.update(acc1, input.size(0))
            avg_acc5.update(acc5, input.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            torch.cuda.synchronize()

            errors = {
                'Epoch': epoch,
                'Train loss': avg_loss.avg.item(),
                'ACC@1': avg_acc1.avg.item(),
                'ACC@5': avg_acc5.avg.item()
            }

            tq_train.set_postfix(errors)

        if self.args.tensorboard:
            log_value('Train_loss', avg_loss.avg, epoch)
            log_value('Train_acc1', avg_acc1.avg, epoch)
            log_value('Train_acc5', avg_acc5.avg, epoch)

        print("Epoch : {}, Train loss : {:.3f}, ACC@1 : {:.3f}, ACC@5 : {:.3f}".format(epoch,
                                                                                       avg_loss.avg.item(),
                                                                                       avg_acc1.avg.item(),
                                                                                       avg_acc5.avg.item()))

        return loss, outputs

    @torch.no_grad()
    def save_prompt(self, epoch):
        # only save the prompt embed if below conditions are satisfied
        if self.args.SAVE_FOR_EACH_EPOCH:
            if self.args.model_type == "vit" and "prompt" in self.args.TRANSFER_TYPE:
                prompt_embds = self.model.enc.transformer.prompt_embeddings.cpu().numpy()
                out = {"shallow_prompt": prompt_embds}

                if self.args.DEEP:
                    deep_embds = self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()
                    out["deep_prompt"] = deep_embds

                torch.save(out, os.path.join(
                    self.args.output_dir, f"prompt_ep{epoch}.pth"))
