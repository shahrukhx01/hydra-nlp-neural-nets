from __future__ import print_function

class BaseLogger:
    def __init__(self, config):
        self.config = config

    def attach(self, model):
        raise NotImplementedError

    def log(self, log_dict):
        raise NotImplementedError
