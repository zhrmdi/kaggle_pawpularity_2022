from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math


def cosine_annealing(lr_max, lr_min, epoch, T_0):
    cosine_term = math.cos(math.pi * epoch / T_0)
    lr = lr_min
    lr_diff = lr_max - lr_min
    if (epoch // T_0) % 2 == 0:
        lr += lr_diff * (1.0 + cosine_term) / 2
    else:
        lr += lr_diff * (1.0 - cosine_term) / 2
    return lr


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, args):
        self.optimizer = optimizer
        self.args = args
        self.epoch = 1
        self._last_lr = 0
        self.T_0 = self.args.T_0
        self.lr_max = self.args.lr
        self.lr_min = self.args.lr_min
        self.init_lr = self.args.init_fc_lr

    def step(self):
        if self.epoch < self.args.fc_epochs:
            lr = cosine_annealing(self.init_lr, self.lr_min, self.epoch, self.T_0 // 2)
        elif self.epoch - self.args.fc_epochs < self.args.warmup_epochs:
            lr = (
                self.args.lr
                * (self.epoch - self.args.fc_epochs)
                / self.args.warmup_epochs
            )
        else:
            epoch = self.epoch - self.args.fc_epochs - self.args.warmup_epochs
            lr = cosine_annealing(self.lr_max, self.lr_min, epoch, self.T_0)

        self._last_lr = lr
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr

        self.epoch += 1

    def get_last_lr(self):
        return [self._last_lr]
