'''
Loss Function Wrapper
Author: Shahrukh Khan(shahrukh.khan3@ibm.com)
'''
from __future__ import print_function
import sys
import torch.nn as nn
from graphs.losses import *

def define_loss(loss_fn):
    if loss_fn in dir(nn):
        return eval('nn.'+loss_fn)()
    else:
        print('Custom Loss')
        try:
            return eval(loss_fn)()
        except Exception as e:
            print('Loss function ' + loss_fn + ' does not exist.')
            sys.exit(-1)
