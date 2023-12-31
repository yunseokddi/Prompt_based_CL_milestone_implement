import math
import torch
import torch.optim as optim

from torch.optim import Optimizer
from typing import Any, Callable, Iterable, List, Tuple, Optional


def create_optimizer(model, args):
    params = []

    for key, value in model.named_parameters():
        if value.requires_grad:
            params.append((key, value))

    if args.WEIGHT_DECAY > 0:
        if args.OPTIMIZER == 'adamw':
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in params
                            if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in params
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.BASE_LR
            )

        else:
            _params = []
            for p in params:
                key, value = p
                lr = args.BASE_LR
                weight_decay = args.WEIGHT_DECAY

                if "last_layer.bias" in key:
                    weight_decay = 0.0

                if args.BIAS_MULTIPLIER == 1.:
                    _params += [{
                        "params": [value],
                        "lr": lr,
                        "weight_decay": weight_decay
                    }]

                else:
                    if "bias" in key and "last_layer.bias" not in key:
                        # use updated lr for this param
                        lr_value = lr * args.BIAS_MULTIPLIER
                    else:
                        lr_value = lr

                    _params += [{
                        "params": [value],
                        "lr": lr_value,
                        "weight_decay": weight_decay
                    }]

            if args.OPTIMIZER == 'adam':
                optimizer = optim.Adam(
                    _params,
                    lr=args.BASE_LR,
                    weight_decay=args.WEIGHT_DECAY,
                )

            else:
                optimizer = optim.SGD(
                    _params,
                    args.BASE_LR,
                    momentum=args.MOMENTUM,
                    weight_decay=args.WEIGHT_DECAY
                )

        return optimizer

    else:
        if args.OPTIMIZER == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.BASE_LR
            )
        else:
            _params = []
            for p in params:
                key, value = p

                lr= args.BASE_LR

                if args.BIAS_MULTIPLIER == 1.:
                    _params += [{
                        "params": [value],
                        "lr": lr,
                    }]

                else:
                    if "bias" in key and "last_layer.bias" not in key:
                        # use updated lr for this param
                        lr_value = lr * args.BIAS_MULTIPLIER

                    else:
                        lr_value = lr

                    _params += [{
                        "params": [value],
                        "lr": lr_value,
                    }]

            optimizer = optim.SGD(
                _params,
                args.BASE_LR,
                momentum=args.MOMENTUM,
            )

        return optimizer

class AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """

    def __init__(
            self,
            params: Iterable,
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True
    ) -> None:
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = {
            "lr": lr, "betas": betas, "eps": eps,
            "weight_decay": weight_decay, "correct_bias": correct_bias
        }
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[Callable]:
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, "
                        "please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(
                        bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        return loss
