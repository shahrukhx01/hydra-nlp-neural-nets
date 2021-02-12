'''
Metrics
Author: Shahrukh Khan(shahrukh.khan3@ibm.com)
'''
from __future__ import print_function
import re
from graphs.metrics import *

class Metrics:
    def __init__(self, config):
        self.config = config
        self.define_metrics()

    def _score_case(self, name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def define_metrics(self):
        if hasattr(self.config.agent, 'metrics'):
            self.metric_list = dict()
            met = self.config.agent.metrics

            for m in met:
                try:
                    self.metric_list[self._score_case(m)] = eval(m)()
                except Exception as e:
                    print('[Error] Undefined Metric: \"' + m + '\" is not a valid metric or is not defined.')
        else:
            raise Exception('[Error] No valid metrics has been defined in configuration')

    def reset(self):
        for name, metric in self.metric_list.items():
            metric.reset()

    def update(self, output, target):
        for name, metric in self.metric_list.items():
            metric.update((output, target))

    def compute(self, mode):
        output_metrics = dict()
        for name, metric in self.metric_list.items():
            if mode+'_'+name not in output_metrics:
                output_metrics[mode+'_'+name] = 0
            output_metrics[mode+'_'+name] = metric.compute()
        return output_metrics
