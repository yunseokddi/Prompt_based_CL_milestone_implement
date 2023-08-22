import torch
import os

from fvcore.common.checkpoint import Checkpointer
from utils.utils import AverageMeter, get_world_size
from tqdm import tqdm

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

            print(input.shape)

            with torch.set_grad_enabled(True):
                outputs = self.model(input)




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