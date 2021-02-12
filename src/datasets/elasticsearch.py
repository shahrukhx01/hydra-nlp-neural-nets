'''
Elasticsearch Data Loader
Author: Shahrukh Khan (shahrukh.khan3@ibm.com)
'''
from __future__ import print_function
import pandas as pd
import numpy as np
from datasets.base import BaseDataLoader
import calendar
import time
import eland as ed
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import math

class ESDataLoader(BaseDataLoader):
    def __init__(self,agent):
        super().__init__(agent)
        """
        This class will load datasets and initiate data preprocessing on datasets
        :param config: Hydra config object
        :param logger: MLFlow logger object
        """

        # Set batch timestamp range
        self.batch_start_dt = self.config.dataloader.batch_range_dt.batch_start_dt
        self.batch_end_dt = self.config.dataloader.batch_range_dt.batch_end_dt

        # Set timestamp columns
        self.inc_processed_path = self.config.dataloader.data_dir.inc_processed_path
        self.chg_processed_path = self.config.dataloader.data_dir.chg_processed_path

         # Set processed indices names
        self.inc_dt_column = self.config.dataloader.batch_range_dt.inc_dt_column
        self.chg_dt_column = self.config.dataloader.batch_range_dt.chg_dt_column

        # Set window size
        self.window_size = self.config.temporal_filtering.window_size
        
        # Instantiate ES client
        self.es = Elasticsearch()

           
        # Load Incident Dataframe
        self.inc_data = None

        # Load Incident Dataframe
        self.chg_data = None
        
        #self.init_data_loader()
        # Run the Change Incident pipeline
        #agent.run()


    def init_data_loader(self):
        """
        This will initiate data loading process
        :param self:
        """       
        incident_ids = list(self.load_data_ids(self.inc_data_path, 'Incident'))
        self.process_batches(incident_ids)


    def load_data_ids(self, path, dataset_name=''):
        """
        This will read data frame from es
        :param path: String path to index in es
        :param dataset_name: String name of the dataset for logging
        :return: Pandas dataframe containing raw data
        """
        df, dt_col = self.load_data(path, dataset_name)

        return ed.eland_to_pandas(df[dt_col]).values

    def load_data(self, path, dataset_name='', batch_start_dt=None, batch_end_dt=None):
        """
        This will read data frame from es
        :param path: String path to index in es
        :param dataset_name: String name of the dataset for logging
        :return: Pandas dataframe containing raw data
        """
        # Read CSV from drive
        self.log.info('Loading {} ids'.format(dataset_name,path))

        if not batch_start_dt:
            batch_start_dt = self.batch_start_dt
        
        if not batch_end_dt:
            batch_end_dt = self.batch_end_dt
        
        # Set start of batch date filter
        start_dt = ''
        id = ''
        dt_col = ''

        if dataset_name == 'Incident':
            start_dt = batch_start_dt
            id = self.config.dataloader.incident_id
            dt_col = self.inc_dt_column

        elif dataset_name == 'Change':
            start_dt = "{}||-{}d".format(batch_start_dt, self.window_size)
            id = self.config.dataloader.change_id
            dt_col = self.chg_dt_column

        df = ed.read_es(self.es, path)            
        df = df.es_query({
            "range": {
                dt_col: {
                    "gte": start_dt,
                    "lte": batch_end_dt
                }
                }
            })

        return (df, dt_col)


    def save_output(self, df):
        """
        This will save data file to elasticsearch output index
        :param df: Dataframe to be saved in ES.
        :return: Pandas dataframe containing raw data
        """
        # Save result to ES
        self.log.info('Saving Output: to {}'.format(self.config.dataloader.output_index))
        df = df.set_index(df.id)
        ed.pandas_to_eland(df, 
                           self.config.dataloader.elasticsearch_host, 
                           self.config.dataloader.output_index, 
                           es_if_exists="append", 
                           es_refresh=True)


    def batch_to_memory(self, batch_start_dt, batch_end_dt, dataset_name, path):
        """
        This will load Eland df to memory using pandas.
        :param batch_start_dt: start of the minibatch date
        :param batch_end_dt: end of the minibatch date
        :param dataset_name: dataset name for logging
        :param path: ES index name
        :return: Pandas dataframe containing raw data
        """
        (df, dt_col) = self.load_data(path, dataset_name, batch_start_dt, batch_end_dt)
        return ed.eland_to_pandas(df)
    


    def process_batches(self, dts):
        """
        This will create minibatches from main batch and process them
        :param dts: list of all dates present in main batch
        """
        dts.sort()
        inc_batch_size = math.ceil(len(dts)* self.config.dataloader.batch_size)
        inc_batches = len(dts) // inc_batch_size + 1

        for inc in range(inc_batches):            
            inc_batch_dts = dts[inc*inc_batch_size:(inc+1)*inc_batch_size]

            if len(inc_batch_dts) < 1:
                break
            
            self.inc_data =  self.batch_to_memory(str(np.min(inc_batch_dts)), 
                                    str(np.max(inc_batch_dts)), 
                                    'Incident', 
                                    self.inc_data_path)
            self.inc_data = self.inc_data.set_index(self.config.dataloader.incident_id)

            self.chg_data = self.batch_to_memory(str(np.min(inc_batch_dts)), 
                                str(np.max(inc_batch_dts)), 
                                'Change', 
                                self.chg_data_path)
            self.chg_data = self.chg_data.set_index(self.config.dataloader.change_id)

            self.init_preprocessing()

            self.agent.run()
           
    