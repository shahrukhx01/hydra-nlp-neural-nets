'''
Text Preprocessor Unit Test
Author: Shahrukh Khan (shahrukh.khan3@ibm.com)
'''
import unittest
import pandas as pd
from pandas._testing import assert_frame_equal
import logging
from pathlib import Path
import os
import sys
import pickle
from box import Box
from test_processor_base import TestPreprocessorBase
from utils.text_preprocessor import TextPreprocessor

class TestTextPreprocessor(TestPreprocessorBase, unittest.TestCase):        
 
    def test_text_preprocessing(self):
        """
        This will validate the test preprocessing pipeline for both sentence and token cleansing
        :param: self
        :return: 
        """
        # Set dummy hydra config 
        config = Box({ # Box library allows dicts keys to be accessed via dot
            'dataloader': {
                'na_value': ''
            }
        })

        # Invoke data preprocessor to get processed dataframe
        self.log.info("Initializing data preprocessor")
        preprocessor = TextPreprocessor(config, None, self.preprocess_config, self.data)

        # Invoke data preprocessor to get processed dataframe
        self.log.info("Preprocess the data")
        processed_data = preprocessor.preprocess_data()

        # Compare the processed data with groundtruth
        self.log.info("Validating the preprocessed data vs groundtruth")
        assert_frame_equal(processed_data, self.groundtruth)

 
if __name__ == '__main__':

    # Init & configure logger
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    log = logging.getLogger('[TEST PREPROCESSOR]')

    # Configure file paths
    log.info("Initializing paths for data files")
    base_path = Path(__file__).parents[3]
    data_path = '{}/data/test_preprocessing_data.csv'.format(base_path)
    groundtruth_path = '{}/data/test_preprocessing_ground_truth.csv'.format(base_path)

    # Load raw data file
    log.info("Loading data file from {}".format(data_path))
    data = pd.read_csv(data_path)
    
    # Load ground truth data file
    log.info("Loading data file from {}".format(groundtruth_path))
    groundtruth = pd.read_csv(groundtruth_path)   

    log.info("Creating configuration for data preprocessing")
    preprocess_config = []
    # Use ground truth to create config for preprocessing
    for column in groundtruth.columns:
        sent_only = False
        # Insert only columns with suffix _processed or _tokens
        if '_processed' in column or '_tokens' in column:
            # Extract column name by removing suffix
            column_extracted = '_'.join(column.split('_')[:2])
            if '_processed' in column:
                sent_only = True
            preprocess_config.append({
                'column': column_extracted,
                'sent_only': sent_only
            })
  
    # Create TestSuite instance and supply params to validate preprocessing pipeline
    log.info("Creating configuration for data preprocessing")
    suite = unittest.TestSuite()
    suite.addTest(TestPreprocessorBase.parametrize(TestTextPreprocessor, preprocess_config=preprocess_config, data=data, groundtruth=groundtruth, log=log))
    unittest.TextTestRunner(verbosity=2).run(suite)