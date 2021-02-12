'''
Butane: MNIST Pytorch Agent
Author: Shahrukh Khan(shahrukh.khan3@ibm.com)
'''
from __future__ import print_function
import sys
import math
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.autograd import Variable

from utils.metrics import *
from agents.base import BaseAgent
from utils.optimizer import Optim
from utils.loss import define_loss
from graphs.models.mnist import MNIST
from utils.lr_scheduler import LRScheduler
from datasets.mnist import MNISTDataLoader

class Agent(BaseAgent):
    def __init__(self, config, logger):
        super().__init__(config, logger)

        # Initialize Model
        self.model = MNIST(config)
        self.logger.log_torch_model(self.model)

        # Initialize Dataloader
        self.data_loader = MNISTDataLoader(config)

        # Define Loss Function
        self.loss = define_loss(config.agent.trainer.loss)

        # Define Metrics List
        self.metrics = Metrics(config)

        # Initialize Optimizer
        self.optimizer = Optim(self.model.parameters(), config)
        self.lr_scheduler = LRScheduler(self.optimizer, config)

    def run(self):
        try:
            if self.config.agent.trainer.enable:
                self.train()
            if self.config.agent.test.enable:
                 self.test()
        except KeyboardInterrupt:
            print('> Ctrl + C Key Press Detected... Waiting to finalize model.')

    def train(self):
        print('> TRAINING MODEL')
        for epoch in tqdm(range(1, self.config.agent.trainer.max_epoch + 1)):
            self.train_one_epoch(epoch)
            self.validate(epoch)

    def train_one_epoch(self, epoch):
        # Enable Training Mode
        self.model.train()
        self.metrics.reset()

        # Iterate Training Process
        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            # data, target = data.to(self.device), target.to(self.device)

            # Compute Loss
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)

            # Backprop Gradient
            loss.backward()
            self.optimizer.step()

            # Append Batch-Wise Metrics
            self.metrics.update(output, target)

        # Compute & Log Epoch-Wise Metrics
        train_metrics = self.metrics.compute('train')
        self.logger.log_metrics(train_metrics)

    def validate(self, epoch):
        pass

    def test(self):
        print('> TESTING MODEL')

    def finalize(self):
        pass
