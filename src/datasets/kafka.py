'''
Kafka Data Loader
Author: Shahrukh Khan
'''
from __future__ import print_function
import pandas as pd
from datasets.base import BaseDataLoader
#from kafka import KafkaConsumer
import json
import calendar
import time
from elasticsearch import Elasticsearch
from elasticsearch import helpers

class KafkaDataLoader(BaseDataLoader):
    def __init__(self, agent):
        super().__init__(agent)
        """
        This class will load datasets and initiate data preprocessing on datasets
        :param config: Hydra config object
        :param logger: MLFlow logger object
        """
        # Initializing a queue for holding minibatches
        self.queue = []

        # Configure elasticsearch
        self.elasticsearch_instance = self.init_es()

    
    def init_es(self):
        """
        This class will initialize `elasticsearch` python client and `es_pandas` client
        :param self:
        :return elasticsearch: Dictionary containing both `elasticsearch` & `es_pandas` clients
        """
        es = Elasticsearch()

        # Create index name based on current timestamp
        es_index = self.config.dataloader.eleasticsearch_index

        # Create index after checking that it doesn't exist already
        if not es.indices.exists(index=es_index):
            es.indices.create(index=es_index)
            self.log.info('Created elasticsearch index {} '.format(es_index))

        return {'es_client': es, 'es_index': es_index}


    def consume_data(self):
        """
        This class will load datasets in the form of mini-batches from `kafka` channel
        :param self:
        """

        consumer = KafkaConsumer(
        'test5',
         bootstrap_servers=['localhost:9092'],
         auto_offset_reset='earliest',
         enable_auto_commit=True,
         group_id='my-group',
         value_deserializer=lambda x: json.loads(x.decode('utf-8')))

        # Parse incoming kafka messages
        for message in consumer:
            message = message.value
            data = json.loads(message)

            # Add new messages to data loader queue
            self.queue.append(data)

            # Initiate the change incident linkage pipeline
            self.init_pipeline()


    def init_pipeline(self):
        """
        Start the change incident linkage pipeline
        for the current mini-batch
        :param path: String path to file on disk
        :return: Pandas dataframe containing raw data
        """
        # Read data from kafka based queue
        self.log.info('Reading data from the kafka based queue')
        self.batch_data = self.get_data()

        # Check if the current batch is populated
        self.log.info('Populating dataframes based on current batch')
        if self.is_batch_empty():
            
            # Load dataframes and preprocess
            self.load_data()           
            self.init_preprocessing()

            # Run the agent to pass batch through the pipeline
            self.log.info('Starting change incident linkage pipeline')
            self.agent.run()


    def load_batch(self):
        """
        This will read a mini-batch from queue
        :param self:
        :return: data in json form
        """
        # Read in memory queue for batch of data
        return self.get_data()


    def load_data(self):
        """
        This function will populate incident and change dataframes with
        mini batches
        :param self:
        """
        # Extract and sort keys
        keys = list(self.batch_data.keys())
        keys.sort()
        chg_key, inc_key = keys

        # Convert json to pandas dataframe for each key
        self.inc_data = pd.DataFrame(self.batch_data[inc_key])
        self.chg_data = pd.DataFrame(self.batch_data[chg_key])


    def is_batch_empty(self):
        """
        This return boolean response based on data existence in current batch
        :param self:
        :param dataset_name: String name of the dataset for logging
        :return data_present: Boolean flag set based on data existence
        """
        # Extract keys
        inc_key, chg_key = list(self.batch_data.keys())

        # Validate if keys and values exist
        data_present = inc_key and chg_key and self.batch_data[inc_key] and self.batch_data[chg_key]
        return data_present


    def get_data(self):
        """
        Start the change incident linkage pipeline
        for the current mini-batch
        :param self: 
        :return data_batch: JSON earliest batch of data present in queue
        """
        data_batch = {}

        # load first batch if the queue isn't empty
        if len(self.queue) > 0:
            data_batch = self.queue.pop(0)

        #return data batch 
        return data_batch

    def save_output(self, actions, dataset_name=''):
        """
        Save output to elasticsearch index 
        for the current mini-batch
        :param self: 
        :param dataframe: Dataframe containing outputs of current batch
        :return:
        """
        # Write the batch output to elasticsearch
        self.log.info('Saving output to elasticsearch index {}'.format(self.elasticsearch_instance['es_index']))
        helpers.bulk(self.elasticsearch_instance['es_client'], actions)