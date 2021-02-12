'''
Learning Rate Scheduler Wrapper
Author: Shahrukh Khan(shahrukh.khan3@ibm.com)
'''
from __future__ import print_function
from torch.optim.lr_scheduler import *

class LRScheduler(object):
    def _makeOptimizer(self):
        if self.method == 'None':
            self.lr_scheduler = None
        elif self.method == 'LambdaLR':
            self.lr_scheduler = LambdaLR(self.optim, self.lr_lambda)
        elif self.method == 'StepLR':
            self.lr_scheduler = StepLR(self.optim, self.step_size, self.gamma)
        elif self.method == 'MultStepLR':
            self.lr_scheduler = MultiStepLR(self.optim, self.step_size, self.gamma)
        elif self.method == 'ExponentialLR':
            self.lr_scheduler = ExponentialLR(self.optim, self.gamma)
        elif self.method == 'CosineAnnealingLR':
            self.lr_scheduler = CosineAnnealingLR(self.optim, self.T_max)
        elif self.method == 'ReduceLROnPlateau':
            self.lr_scheduler = ReduceLROnPlateau(self.optim)
        elif self.method == 'CyclicLR':
            self.lr_scheduler = CyclicLR(self.optim, self.base_lr, self.max_lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, optim, config):
        self.optim = optim
        self.config = config

        self.method = config.agent.lr_scheduler.method

        self._makeOptimizer()

    def step(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
