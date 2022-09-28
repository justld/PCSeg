# import torch
# from torch.optim import lr_scheduler

import paddle
from paddle.optimizer.lr import LRScheduler


class WarmUpLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr, warmup_steps, momentum, decay):
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.momentum = momentum
        self.decay = decay

        if self.warmup_steps < 1:
            self.warmup_steps = 1

        # cyclic lr
        # self.initial_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
        #                                          base_lr=0,
        #                                          max_lr=self.lr,
        #                                          step_size_up=self.warmup_steps,
        #                                          step_size_down=self.warmup_steps,
        #                                          cycle_momentum=False,
        #                                          base_momentum=self.momentum,
        #                                          max_momentum=self.momentum)
        # self.initial_scheduler = paddle.optimizer.lr.

        # our params
        self.last_epoch = -1  # fix for pytorch 1.1 and below
        self.finished = False  # am i done
        super().__init__(optimizer)

    def get_lr(self):
        return [self.lr * (self.decay**self.last_epoch) for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished or self.initial_scheduler.last_epoch >= self.warmup_steps:
            if not self.finished:
                self.base_lrs = [self.lr for lr in self.base_lrs]
                self.finished = True
            return super(WarmUpLR, self).step(epoch)
        else:
            return self.initial_scheduler.step(epoch)
