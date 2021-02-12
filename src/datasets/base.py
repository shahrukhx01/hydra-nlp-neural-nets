'''
Base Data Loader
Author: Shahrukh Khan(shahrukh.khan3@ibm.com)
'''
from __future__ import print_function
from abc import ABC, abstractmethod
import logging
import os
from utils.text_preprocessor import TextPreprocessor
import calendar
import time

class BaseDataLoader(ABC):

    def __init__(self, agent):
        """
        Parent class for all dataloading activities
        :param self:
        :param config: Hydra configuration object
        """
        self.config = agent.config
        self.logger = agent.logger
        self.agent = agent

        # Initialize Dataloader Parameters
        self.inc_data_path = self.config.dataloader.data_dir.inc_path
        self.chg_data_path = self.config.dataloader.data_dir.chg_path
        self.ignore_errors = self.config.dataloader.ignore_errors
        self.inc_data_config = eval(str(self.config.dataloader.data_config.incident))
        self.chg_data_config = eval(str(self.config.dataloader.data_config.change))
        self.inc_data = None
        self.chg_data = None

        # Init & configure logger
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
        self.log = logging.getLogger('DATA LOADER') 

 
    def init_preprocessing(self):
        """
        This will iniate preprocessing on datasets
        :param self:
        :return:
        """
        # Preprocess Incident Data
        self.log.info('Performing data preprocessing pipeline on Incident')
        self.inc_data = self.preprocess_data(self.get_inc_config(), 
                                            self.get_inc_data(),
                                            self.config.dataloader.incident_id)


        # Preprocess Change Data
        self.log.info('Performing data preprocessing pipeline on Change')
        self.chg_data = self.preprocess_data(self.get_chg_config(), 
                                            self.get_chg_data(), 
                                            self.config.dataloader.change_id)
    

    def preprocess_data(self, data_config, data, id=None):
        """
        This will remove non_ascii characters from the document
        :param data_config: List of dataset column configurations
        :param data: Pandas dataframe containing raw data
        :return: Pandas dataframe containing processed data
        """
        # Perform text preprocessing
        preprocessor = TextPreprocessor(self.config, self.logger, data_config, data, id)
        return preprocessor.preprocess_data()

    
    def get_inc_data(self):
        """
        This will return Incident dataframe
        :param self: 
        :return: Pandas dataframe containing incident data
        """
        return self.inc_data


    def get_chg_data(self):
        """
        This will return Change dataframe
        :param self: 
        :return: Pandas dataframe containing change data
        """
        return self.chg_data


    def set_inc_data(self, inc_data):
        """
        This will update Incident dataframe
        :param self: 
        :param inc_data: Updated dataframe
        :return: Pandas dataframe containing incident data
        """
        self.inc_data = inc_data


    def set_chg_data(self, chg_data):
        """
        This will update Change dataframe
        :param self: 
        :param inc_data: Updated dataframe
        :return: Pandas dataframe containing incident data
        """
        self.chg_data = chg_data

    
    def get_inc_config(self):
        """
        This will return Incident configurations
        :param self: 
        :return: Pandas dicts containing incident column configurations
        """
        return self.inc_data_config

    
    def get_chg_config(self):
        """
        This will return Change configurations
        :param self: 
        :return: List of dicts containing change column configurations
        """
        return self.chg_data_config


    @abstractmethod
    def load_data(self, path, dataset_name=''):
        raise NotImplementedError

    @abstractmethod
    def save_output(self, dataframe, dataset_name=''):
        raise NotImplementedError