'''
Domain specific entity extraction inference
Author: Shahrukh Khan
'''
from __future__ import print_function
import spacy

def get_model(pattern_ner):
    """
    This class will load trained spacy ner model from disk
    :param self:
    :return:
    """
    # Load model from disk
    pattern_ner.log.info("Loading domain ner model from {}".format(pattern_ner.model_path))
    nlp = spacy.load(pattern_ner.model_path)
    return nlp
