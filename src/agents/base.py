'''
Butane: Base Agent Abstraction
Author: Shahrukh Khan (shahrukh.khan3@ibm.com)
'''
from __future__ import print_function
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, config, logger):
        """
        Parent class for all agents that implement the core logic
        :param self:
        :param config: Hydra configuration object
        """
        # Initialize Config and Logger
        self.config = config
        self.logger = logger
    
    def run(self):
        raise NotImplementedError

    def run_explicit_linkage(self):
        raise NotImplementedError

    def extract_domain_ner(self):
        raise NotImplementedError

    def run_text_similarity(self):
        raise NotImplementedError
