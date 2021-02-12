from __future__ import print_function
import os
import json
from pprint import pprint
from easydict import EasyDict

def get_config(json_file):
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError as e:
            print("Invalid JSON File: " + e)
            exit(-1)

def save_config(fname, json_file):
    out = open(fname + '/config.json', 'w')
    out.write(json.dumps(json_file))
    out.close()

def disp_config(config):
    try:
        # Display Experiment Information
        print('='*80)
        print('Experiment Name: ' + config.exp_name)
        print('-'*80)
    except AttributeError as ex:
        print("Missing parameter in config: exp_name")
        print(ex)
        exit(-1)

    # Display Configuration Setting
    print("[Configuration Parameters]\n")
    pprint(config)
    print('-'*80)

def parse_config(json_file):
    # Get Configuration File Contents
    config, config_dict = get_config(json_file)

    # Check Configuration Type
    if hasattr(config, 'base_config'):
        raise NotImplementedError('Multi-Parameter Mode Currently Not Implemented')
    else:
        disp_config(config)
