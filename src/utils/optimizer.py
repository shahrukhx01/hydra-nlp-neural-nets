import math
import torch.optim as optim

class Optim(object):
    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, lr_decay=self.lr_decay, weight_decay=self.weight_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, betas=self.beta, eps=self.eps, weight_decay=self.weight_decay)
        elif self.method == 'adamw':
            self.optimizer = optim.AdamW(self.params, lr=self.lr, betas=self.beta, eps=self.eps, weight_decay=self.weight_decay)
        elif self.method == 'sparseadam':
            self.optimizer = optim.SparseAdam(self.params, lr=self.lr, betas=self.beta, eps=self.eps)
        elif self.method == 'adamax':
            self.optimizer = optim.Adamax(self.params, lr=self.lr, betas=self.beta, eps=self.eps, weight_decay=self.weight_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    # def __init__(self, params, method, lr, max_grad_norm, lr_decay=1, start_decay_at=None):
    def __init__(self, params, config):
        self.params = list(params)

        # Initialize Optimization Method
        self.method = config.agent.optim.method

        # Initialize Optimization Parameters
        self.lr = config.agent.optim.lr
        self.lr_decay = config.agent.optim.lr_decay
        self.momentum = config.agent.optim.momentum
        self.weight_decay = config.agent.optim.weight_decay
        self.beta = (config.agent.optim.beta1, config.agent.optim.beta2)
        self.amsgrad = config.agent.optim.amsgrad
        self.max_grad_norm = config.agent.optim.clip
        self.eps = config.agent.optim.eps

        self._makeOptimizer()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        for param in self.params:
            grad_norm += math.pow(param.grad.data.norm(), 2)

        grad_norm = math.sqrt(grad_norm)
        if grad_norm > 0:
            shrinkage = self.max_grad_norm / grad_norm
        else:
            shrinkage = 1.

        for param in self.params:
            if shrinkage < 1:
                param.grad.data.mul_(shrinkage)

        self.optimizer.step()
        return grad_norm
