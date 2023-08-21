import torch

from fvcore.common.checkpoint import Checkpointer

class Trainer(object):
    def __init__(self, model, criterion, data_loader, optimizer, lr_scheduler, device, args):
        self.model = model
        self.criterion = criterion
        self.data_loader = data_loader
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
