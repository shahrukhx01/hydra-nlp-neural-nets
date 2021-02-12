'''
Data Loader Unit Test
Author: Shahrukh Khan
'''
import unittest
import pandas as pd
from pandas._testing import assert_frame_equal
import logging
from pathlib import Path
import os
import sys
from test_dataloader_base import TestDataLoaderBase
from datasets.servicenow import SNDataLoader
import pickle
from pprint import pprint

class TestDataLoader(TestDataLoaderBase, unittest.TestCase):        
 
    def test_dataloader(self):
        """
        This will validate the test data loading and preprocessing pipeline for both sentence and token cleansing
        :param: self
        :return: 
        """

        # Invoke data preprocessor to get processed dataframe
        self.log.info("Initializing data loader")
        data_loader = SNDataLoader(self.config, None)

        # Add processed columns mappings
        chg_columns = ['change_abstract_tokens', 'change_description_tokens', 
                      'change_abstract_processed', 'change_description_processed']
        
        # Compare the processed data with groundtruth
        self.log.info("Validating the loaded preprocessed data vs groundtruth")
        assert_frame_equal(data_loader.get_chg_data()[chg_columns], self.groundtruth[chg_columns])

 
if __name__ == '__main__':

    # Init & configure logger
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    log = logging.getLogger('[TEST DATA LOADER]')

    # Configure file paths
    log.info("Initializing paths for data files")
    base_path = Path(__file__).parents[3]
    groundtruth_path = '{}/data/test_preprocessing_ground_truth.csv'.format(base_path)

    
    # Load ground truth data file
    log.info("Loading groundtruth data from {}".format(groundtruth_path))
    groundtruth = pd.read_csv(groundtruth_path)

    
    # Load data loader config
    log.info("Loading configurations for data loading")
    config_pickle_path = '{}/data/data_loader_config.pickle'.format(base_path)
    config_pickle_file = open(config_pickle_path,'rb')
    config = pickle.load(config_pickle_file)
    config_pickle_file.close()

    #Log data loader config
    log.info("Loaded the following data loader config")
    pprint(config)
    
    # Create TestSuite instance and supply params to validate preprocessing pipeline
    log.info("Starting Data Loader Unit Test")
    suite = unittest.TestSuite()
    suite.addTest(TestDataLoaderBase.parametrize(TestDataLoader, config=config, groundtruth=groundtruth, log=log))
    unittest.TextTestRunner(verbosity=2).run(suite)