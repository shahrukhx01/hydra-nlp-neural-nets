'''
Change Incident Linkage
Author: Shahrukh Khan (shahrukh.khan3@ibm.com)
'''
from __future__ import print_function
from agents.base import BaseAgent
from datasets.servicenow import SNDataLoader
from core.domain_ner.pattern_based_ner import PatternNer
from core.temporal_filtering.time_window import TimeWindowFilter
from core.text_similarity.text_similarity_trial import TextSimilarityTrial

class Agent(BaseAgent):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        
        # Initialize properties
        self.data_loader = SNDataLoader(self)
        
        self.text_similarity_computer = None

        # Run kafka consumer
        #if config.dataloader.data_loader_type == 'elasticsearch':
        #self.data_loader.init_data_loader()
            
    
    def run(self):
        # Prepare similarity corpus
        self.text_similarity_computer = TextSimilarityTrial(self)

        # Extract domain ner
        self.extract_domain_ner()

        # Run time window filtering
        self.run_time_window_filter()

    def extract_domain_ner(self):
        ner_pipeline = PatternNer(self)
        ner_pipeline.run()


    def run_time_window_filter(self):
        time_window_filter = TimeWindowFilter(self)
        time_window_filter.run()