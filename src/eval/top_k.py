'''
Top K Evaluation
Author: Sanjana Sahayaraj(sanjana@ibm.com)
Co Author: Shahrukh Khan(shahrukh.khan3@ibm.com)
'''
from __future__ import print_function
import logging
import os
from collections import namedtuple
import time
from tqdm import tqdm, tqdm_pandas
import pandas as pd
import numpy as np
#import eland as ed
#from elasticsearch import Elasticsearch
from matplotlib import pyplot as plt
import seaborn as sns
#import spacy
import re

from text_similarity_trial import TextSimilarityTrial
#tqdm.pandas(tqdm())

class TopK:
    def __init__(self):
        """
        Class for evaluation incident, change pairs against ground truth
        :param self:
        """

        # ****************** CONFIGURATION START # ****************** 


        # init column mapping
        self.columns = {
            'inc_id': 'incident_id',
            'chg_id': 'change_id',
            'score': 'similarity_score'
            }
        
        # data paths
        self.predictions_path = 'change_incident_out1'
        self.groundtruth_path = 'groundtruth'

        # ES client
        self.es = Elasticsearch()

        # set data source
        self.data_source = 'es' # i.e file, es etc.

        # ****************** CONFIGURATION END # ****************** 

        # init data list variables
        self.gt_ic_pairs = set()
        self.ranking_tuples = list()
        
        #create named tuples
        self.incident_score_entry = namedtuple("incident_score_entry",[self.columns['chg_id'],self.columns['score']])
        self.ic_pair = namedtuple("ic_pair",[self.columns['inc_id'], self.columns['chg_id']])
        self.ranking_entry = namedtuple("ranking_entry",[self.columns['inc_id'], self.columns['chg_id'],self.columns['score']])

        # Init & configure logger
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
        self.log = logging.getLogger('TOPK EVALUATION') 

        # load data
        self.load_data()

        # generate data tuple lists
        self.generate_tuple_lists()


    def load_data(self):
        """
        This will load data to memory
        :param self:
        :return:
        """
        if self.data_source == 'file':
            self.load_data_files()
        elif self.data_source == 'es':
            self.load_data_es()

    def load_data_es(self):
        """
        This will load data to memory from ES
        :param self:
        :return:
        """
        self.gt_df = ed.eland_to_pandas(ed.read_es(self.es, self.groundtruth_path))[[self.columns['inc_id'], 
                                                        self.columns['chg_id']]]
        self.pred_df = ed.eland_to_pandas(ed.read_es(self.es, self.predictions_path))[[self.columns['inc_id'], 
                                                        self.columns['chg_id'],
                                                        self.columns['score']]]
        self.pred_df.rename({
            'overall_score': 'similarity_score'
        })
    def load_data_files(self):
        """
        This will load data to memory from files
        :param self:
        :return:
        """
        self.log.info('Loading datasets from files.')

        # load groundtruth
        self.gt_df = pd.read_csv(self.groundtruth_path)[[self.columns['inc_id'], 
                                                        self.columns['chg_id']]]
        self.gt_df = self.gt_df.drop_duplicates()

        # load predictions
        self.pred_df = pd.read_csv(self.predictions_path)[[self.columns['inc_id'], 
                                                        self.columns['chg_id'],
                                                        self.columns['score']]]

        self.gt_df[self.columns['chg_id']] =  self.gt_df[self.columns['chg_id']].str.upper() # remove later
        #self.pred_df = self.pred_df.drop_duplicates()

    def tuple_sorter(self, tuple_item):
        """
        This will sort the tuples
        :param self:
        :return:
        """
        return tuple_item[1]

    def generate_tuple_lists(self):
        self.log.info('Generating IC pairs for Groundtruth')
        for index, row in tqdm(self.gt_df.iterrows()):
            self.gt_ic_pairs.add(self.ic_pair(incident_id=row[self.columns['inc_id']], 
                                              change_id=row[self.columns['chg_id']]))

        self.log.info('Generating IC pairs for predictions')
        for index, row in tqdm(self.pred_df.iterrows()):
            self.ranking_tuples.append(self.ranking_entry(incident_id=row[self.columns['inc_id']], 
                                       change_id=row[self.columns['chg_id']], 
                                       similarity_score=row[self.columns['score']]))
    
    def generate_inc_score_dict(self):
        """
        This will generate all candidate incident change pairs.
        :param self:
        :return:
        """
        incident_score_dict = dict()
        incident_gold_entry_dict = dict()
        self.scores_dict = {'gold':[], 'non_gold':[]}

        for rank_entry in self.ranking_tuples:
            inc_id = rank_entry.incident_id

            if(inc_id in incident_score_dict):
                incident_score_dict[inc_id].append(self.incident_score_entry(change_id = rank_entry.change_id, 
                                                   similarity_score = rank_entry.similarity_score))
            else:
                incident_score_dict[inc_id] = [self.incident_score_entry(change_id = rank_entry.change_id, 
                                                                    similarity_score = rank_entry.similarity_score)]

            if(self.ic_pair(incident_id = inc_id, change_id = rank_entry.change_id) in self.gt_ic_pairs):
                # add gold pred score
                self.scores_dict['gold'].append(rank_entry.similarity_score)
                if(inc_id in incident_gold_entry_dict):
                    incident_gold_entry_dict[inc_id].append(self.incident_score_entry(change_id = rank_entry.change_id, 
                                                            similarity_score = rank_entry.similarity_score))
                else:
                    incident_gold_entry_dict[inc_id] = [self.incident_score_entry(change_id = rank_entry.change_id, 
                                                                             similarity_score = rank_entry.similarity_score)]
            else:
                self.scores_dict['non_gold'].append(rank_entry.similarity_score)
                #incident_gold_entry_dict[inc_id] = None
        
        scores_per_incident_dist = []
        # Sort all scores of an incident with linked incidents
        for inc_id in incident_score_dict:
            incident_score_dict[inc_id] = sorted(incident_score_dict[inc_id], key = self.tuple_sorter, reverse = True)
            scores_per_incident_dist.append(len(incident_score_dict[inc_id]))
        
        return (incident_score_dict, incident_gold_entry_dict, scores_per_incident_dist)

    def plot_scores_dist(self):
        bins = np.linspace(-1, 1, 100)    
        
        sample_len = len(self.scores_dict['non_gold'])
        if len(self.scores_dict['gold']) < len(self.scores_dict['non_gold']):
            sample_len = len(self.scores_dict['gold'])
        non_gold = np.random.choice(np.array(self.scores_dict['non_gold']), sample_len)
        
        sns.kdeplot(self.scores_dict['gold'], shade=True, label='gold')
        sns.kdeplot(non_gold, shade=True, label='non_gold')
        plt.legend(loc='upper right')
        plt.show()
        

    def compute_topk_score(self, k=1):
        """
        This will generate all candidate incident change pairs.
        :param self:
        :return:
        """
        self.log.info('Computing incident score dictionary')
        incident_score_dict, incident_gold_entry_dict, scores_per_incident_dist = self.generate_inc_score_dict()

        topk_count = 0
        topk_fp_count = 0
        total_count = 0

        max_entries = max(scores_per_incident_dist)
        for inc_id in incident_score_dict:
            score_entries = incident_score_dict[inc_id]
            score_len = len(score_entries)
            gold_entry = None
            if inc_id in incident_gold_entry_dict:
                gold_entry = incident_gold_entry_dict[inc_id]
            else:
                continue
            topk_entry = score_entries[:k]

            if gold_entry != None:
                if(len(topk_entry) >= 1):
                    topk_count += len(list(set(gold_entry) & set(topk_entry)))
                    topk_fp_count += len(list(set(gold_entry) - set(topk_entry)))
                
            total_count+=len(score_entries)
        topk_score = (topk_count/total_count) * 100

        return topk_score, total_count

if __name__ == "__main__":
    """ topk_eval = TopK()
    
    for k in range(1,6):
        score_topk, total_count = topk_eval.compute_topk_score(k)
        topk_eval.log.info('score for top{}: {} with total pairs {} '.format(k, score_topk, total_count))

    topk_eval.plot_scores_dist() """

    inc_df = pd.read_csv('/Users/shahrukh/Documents/CH_INC/data/incidents_processed.csv')
    chg_df = pd.read_csv('/Users/shahrukh/Documents/CH_INC/data/changes_processed.csv')
    gt_df = pd.read_csv('/Users/shahrukh/Documents/CH_INC/data/explicit_linkage_gt_overall_score.csv')
    rand_df = pd.read_csv('/Users/shahrukh/Documents/CH_INC/data/explicit_linkage_rand_overall_score_v1.1.csv')
    
    rand_scores = rand_df.res_norm.values

    sns.kdeplot(gt_df.text_sim_score.values, shade=True, label='gold')
    sns.kdeplot(rand_scores, shade=True, label='random')
    plt.legend(loc='upper right')
    plt.show()
    
    gt_scores = np.array(gt_df.overall_score.values)
    rand_scores = np.array(rand_scores)
    print('gt greater than 0.5 {} out of {}'.format(sum(gt_scores>0.5), len(gt_scores)))
    print('rand_scores greater than 0.5 {} out of {}'.format(sum(rand_scores>0.5), len(rand_scores)))
    #print(gt_df.shape)
    #print(inc_df.loc[10])
    

    ## COMPUTATIONAL BLOCK
    """ inc_df.incident_description = inc_df.incident_description.astype(str)
    chg_df.change_description = chg_df.change_description.astype(str)
    text_sim = TextSimilarityTrial(inc_df.incident_description.head(1000), chg_df.change_description.head(1000))
    res = []
    ent_res = []
    for i in tqdm(range(100000)):
        inc = inc_df.sample(n=1)
        chg = chg_df.sample(n=1)
        inc_ents = set( [x for x in str(inc.ents.values[0]).split('||') if x not in ['','nan']])
        chg_ents = set([x for x in str(chg.ents.values[0]).split('||') if x not in ['','nan']])
        denom = len(inc_ents.union(chg_ents))
        if denom <= 0:
            denom = 1
        ent_res.append(len(inc_ents.intersection(chg_ents))/denom)
        res.append(text_sim.compute_similarity(str(inc.text.values[0]).lower(), str(inc.text.values[0]).lower())['similarity_score']) #print(r.change_id, r.incident_id)
    res_norm = [(x-min(res))/(max(res)-min(res)) for x in res]
    overall_score = np.array(res_norm + ent_res)
    overall_score /= 2
    results = []

    for i in range(100000):

        results.append({
            'res_norm': res_norm[i],
            'overall_score': overall_score[i]
        })

    pd.DataFrame(results).to_csv('/Users/shahrukh/Documents/CH_INC/data/explicit_linkage_rand_overall_score_v1.1.csv')
    #gt_df['text_sim_score'] =  res_norm """
    
    
    
