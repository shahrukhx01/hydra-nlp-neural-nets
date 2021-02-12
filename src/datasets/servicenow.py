'''
ServiceNow Data Loader
Author: Shahrukh Khan
'''
from __future__ import print_function
import pandas as pd
from datasets.base import BaseDataLoader
import calendar
import time

class SNDataLoader(BaseDataLoader):
    def __init__(self,agent):
        super().__init__(agent)
        """
        This class will load datasets and initiate data preprocessing on datasets
        :param config: Hydra config object
        :param logger: MLFlow logger object
        """

        # Load Incident Dataframe
        self.inc_data = self.load_data(self.inc_data_path, 'Incident')

        # Load Incident Dataframe
        self.chg_data = self.load_data(self.chg_data_path, 'Change')

        # Initiate Text preprocessing
        self.init_preprocessing()

        # Run the Change Incident pipeline
        #agent.run()


    def load_data(self, path, dataset_name=''):
        """
        This will read data file from disk
        :param path: String path to file on disk
        :param dataset_name: String name of the dataset for logging
        :return: Pandas dataframe containing raw data
        """
        # Read CSV from drive
        self.log.info('Loading {} Dataset: from {}'.format(dataset_name,path))
        return pd.read_csv(path)


    def save_output(self,df, file_name=str(calendar.timegm(time.gmtime()))):
        """
        This will save data file from disk
        :param file_name: String name of the file for logging
        :return: Pandas dataframe containing raw data
        """
        # Save csv to drive
        self.log.info('Saving Output: to {}{}'.format(self.config.dataloader.output_dir,file_name))
        df.to_csv('{}{}.csv'.format(self.config.dataloader.output_dir,file_name))