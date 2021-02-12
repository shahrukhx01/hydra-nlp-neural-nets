'''
Base Text Similiraity Computer
Author: Sanjana Sahayaraj(sanjana@ibm.com)
Co Author: Shahrukh Khan(shahrukh.khan3@ibm.com)
'''
from abc import ABC, abstractmethod
import logging
import os

class TextSimilarityBase(ABC):
    def __init__(self, incident_texts, change_texts):
        """
        Text Similarity Parent Class to compute fuzzy similarity between 
        change description and incident abstract 
        :param self:
        :param config: Hydra configuration object
        """
        # Initialize config and other properties
        #self.config = agent.config
        #self.logger = agent.logger
        #self.data_loader = agent.data_loader
        self.incident_texts = incident_texts#agent.data_loader.get_inc_data()[self.config.text_similarity_fuzzy.incident_field]
        self.change_texts = change_texts  #agent.data_loader.get_chg_data()[self.config.text_similarity_fuzzy.change_field]

        # Init & configure logger
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
        self.log = logging.getLogger('TEXT SIMILARITY COMPUTATION')  
        
    
    @abstractmethod
    def compute_similarity(self, incident_text, change_text):
        raise NotImplementedError