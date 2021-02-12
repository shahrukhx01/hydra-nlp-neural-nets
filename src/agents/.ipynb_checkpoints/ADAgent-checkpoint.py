'''
Enpoint Anomaly Detection Agent
Author: Shahrukh Khan(shahrukh.khan3@ibm.com)
'''
from __future__ import print_function
import os
import sys
import numpy as np

from agents.base import BaseAgent
from graphs.models.epad import get_model
from datasets.servicenow import SNDataLoader

class Agent(BaseAgent):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        
        # Initialize Dataloader
        self.data_loader = SNDataLoader(config)
        
        # Initialize Model
        self.model = get_model(config)
    
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

    def validate(self, epoch):
        pass
    
    def test(self):
        print('> TESTING MODEL')
    
    def finalize(self):
        pass