'''
Text Similarity Trial
Author: Sanjana Sahayaraj(sanjana@ibm.com)
Co Author: Shahrukh Khan(shahrukh.khan3@ibm.com)
'''
from base import TextSimilarityBase
from fuzzywuzzy import fuzz
from pandas import read_csv
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from itertools import islice
from textblob import TextBlob
from collections import Counter
import re
import math
import gensim
from gensim.models import Word2Vec 
from gensim.models import FastText 
import gensim.downloader as api
import statistics
from matplotlib import pyplot as plt
import pandas as pd


gtIds = set()
gtChgs = set()
gtIncs = set()
nonGtIds = set()
nonGtChgs = set()
nonGtIncs = set()

"""
Text Similarity Class to compute fuzzy similarity between change description and incident abstract 
"""
class TextSimilarityTrial(TextSimilarityBase):
	def __init__(self, incident_texts, change_texts):
		super().__init__(incident_texts, change_texts)
		"""
		This class will use combination of fuzzy search and 
		jaccard distance to compute similarity between two texts
		:param self:
		"""
		self.log.info('Initializing Text Similarity Pipeline')
		self.to_remove_entities = self.get_remove_entities()
		self.embed_model = FastText() # needs to be global

	"""
	Utility function to preprocess a given sentence to remove repeated entities
	"""
	def remove_entities(self, sentence, entities):
		# Tokenize the text to create a set of unique words
		tokens = [word.lower() for word in sentence.split()]
		tokens = set(tokens)

		# Remove words in the remove list obtained from tf idf
		tokens = [x for x in tokens if not x in entities]

		return ' '.join(list(set(tokens)))
	
	"""
	Multi dimensional scoring function to score incident change pairs
	Similarity score based on 
	1. Word / Token level cosine similarity for embeddings while ignoring domain heavy terms in this computation
	2. Sentence level word mover's distance similarity computation. The word mover's distance also handles the case of varying lengths of the sentences being used in similarity computation
	3. Fuzzy matching ratio to include character level similarities. In addition penalize for greater difference between lengths of change and incident text.
	4. Items 1 to 3 are symmetric distance measures meaning if the order of change and incident texts are swapped, the score would be the same
	5. Normalize the score obtained to fit in range 0 to 1. This means in the absence of a good pair of (incident, change) text a bad pair can end up getting a good score. In the presence of a good pair, a bad pair will have a low score
	"""
	def compute_similarity(self, incident_text, change_text):
		scores = []
		text_similarity_ranking_list = []

		sim_score = 0
		# Sentence level similarity computation
		sent_score = self.embed_model.wv.wmdistance(incident_text, change_text)
		sim_score -= max(sent_score, 0.0) ## needs to be adjusted

		# Token level similarity computation
		word_score = 0
		for word1, word2 in zip(word_tokenize(incident_text), word_tokenize(change_text)):
			if(word1 in self.to_remove_entities or word2 in self.to_remove_entities):
				word_score += 0
			else:
				word_score += self.embed_model.wv.similarity(word1, word2)
		sim_score += word_score
		
		# Fuzzy matching ratio
		char_score = 0
		char_score += fuzz.token_set_ratio(incident_text, change_text) / 1 + abs(len(incident_text)-len(change_text))/1000
		char_score /= 100 # normalizing fuzzy score -- verification needed.
		sim_score += char_score
		# Enter valid entries into lists
		if(sim_score == float('inf') or sim_score == float('-inf')):
			sim_score = 0

						
		return {
			"similarity_score": sim_score/3,
			"word_sim_score": word_score,
			"char_sim_score": char_score,
			"sent_sim_score": sent_score
		}

	def get_remove_entities(self):
		"""
        Prepares list of entities that shouldn't be present in text
        :param self:
		:return to_remove_entities: List of strings
        """
		# Create word frequency dictionary for words in the corpus	
		self.log.info('Computing word frequency for all words across corpus')

		word_dicts = self.incident_texts.apply(self.create_word_dict).tolist()
		word_dicts.extend(self.change_texts.apply(self.create_word_dict).tolist())
	
		# Compute inverse document frequencies and sort
		idfs = self.compute_idf(word_dicts)
		sorted_idfs = sorted(idfs.items(), key=lambda kv: kv[1])

		# Take out the words to be removed from sorted idfs
		self.to_remove_entities = list(list(zip(*list(islice(sorted_idfs, 10))))[0])

		return self.to_remove_entities


	"""
	Sorting helper function to return the key to sort by
	"""
	def tupleSorter(self, tupleItem):
		return tupleItem[0]

	"""
	Function to get ngrams of sentence based on n specified
	"""
	def get_sentence_ngrams(self, sent, n):
		if not sent: return None
		tb = TextBlob(sent)
		ng = (ngrams(x.words, n) for x in tb.sentences if len(x.words) > n)
		return [item for sublist in ng for item in sublist]

	def cosine_similarity_ngrams(self, a, b):
		vec1 = Counter(a)
		vec2 = Counter(b)
		
		intersection = set(vec1.keys()) & set(vec2.keys())
		numerator = sum([vec1[x] * vec2[x] for x in intersection])

		sum1 = sum([vec1[x]**2 for x in vec1.keys()])
		sum2 = sum([vec2[x]**2 for x in vec2.keys()])
		print("Cosine sums: ", sum1, sum2)
		denominator = math.sqrt(sum1) * math.sqrt(sum2)

		if not denominator:
			print("no denominator")
			return 0.0
		return float(numerator) / denominator

	"""
	Distance metric function to return jaccard distance between 2 sets of tuples
	"""	
	def jaccard_distance(self, a, b):
		a = set(a)
		b = set(b)
		jaccard = 0
		if(len(a)+len(b) != 0):
			jaccard = 1.0 * len(a&b)/len(a|b)
			# print("Jaccard computation: ", jaccard)
		return jaccard

	"""
	Utility Function to help with TF-IDF
	"""
	def create_word_dict(self, sentence):
		countDict = {}
		words = word_tokenize(sentence)
		for word in words:
			if word in countDict:
				countDict[word] += 1
			else:
				countDict[word] = 1
		return countDict

	"""
	Function to compute IDF
	"""
	def compute_idf(self, docList):
		idfDict = {}
		N = len(docList)
		
		for doc in docList:
			for word, val in doc.items():
				if val > 0:
					if(word in idfDict):
						idfDict[word] += 1
					else:
						idfDict[word] = 1
		
		for word, val in idfDict.items():
			idfDict[word] = math.log10(N / float(val))
			
		return idfDict
