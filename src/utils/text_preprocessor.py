'''
Text Preprocessor
Author: Shahrukh Khan
'''
import os
import logging
import re
import pandas as pd
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import eland as ed



class TextPreprocessor:
    def __init__(self, config, logger, preprocess_config, data, id=None):
        """
        This class will preprocess text documents by performing cleansing techniques like tokenization, 
        lemmatization, stop word removal, digit-token removal etc.
        :param preprocess_config: Array of dicts containing column name and sentence only flags
        :param data: Dataframe to be cleansed
        """
         # Init & configure logger
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
        self.log = logging.getLogger('DATA PREPROCESSOR')

        self.na_value = config.dataloader.na_value
        self.preprocess_config = preprocess_config
        self.data = data

       
        self.stoplist = stopwords.words('english')
        self.lmtzr = WordNetLemmatizer()
        


    def remove_non_asciichar(self, sentence):
        """
        This will remove non_ascii characters from the document
        :param sentence: A string
        :return sentence: A string
        """
        sentence = re.sub(r'[^\x00-\x7F]+', " ", " " + str(sentence) + " ").strip()
        return sentence


    def remove_specialchar(self, sentence):
        """
        This will remove any special characters in the document and replace them blank space
        :param sentence: A string
        :return sentence: A string
        """
        replace_list = ('.', '_', '/', '\\', '|', '(', ')', '[', ']', '{', '}', '#', '$', '*', 
                        '@', '%', '&', '+', '>', '<', ';',':', '=', ',' , '!', '?', "'",'"', '`')
        for r in replace_list:
            sentence = sentence.replace(r, ' ')
        return sentence


    def remove_numbers(self, sentence):
        """
        This will remove tokens with digits only
        :param sentence: A string
        :return sentence: A string
        """
        sentence = re.sub(r'\b[0-9]{1,6}\b\W*', ' ', " " + str(sentence) + " ").strip()
        return sentence


    def preprocess_sentence(self, sentence):
        """
        This will preprocess the sentence by removing non_ascii, special characters and digit tokens
        :param sentence: A string
        :return sentence: A string
        """
        # Transform text to lower
        sentence = sentence.lower()

        # Remove non-ascii characters
        sentence = self.remove_non_asciichar(sentence)

        # Remove special characters
        sentence = self.remove_specialchar(sentence)

        # Remove numbers
        sentence = self.remove_numbers(sentence)

        return sentence


    def lematize_tokens(self, sentence):
        """
        This will convert a document to tokens and lemmatize each token
        :param sentence: A string
        :return tokens: A string
        """
        
        # Tokenize the text to create a list of unique words
        tokens = [word.lower() for word in sentence.split()]

        # Remove stopwords
        tokens =  [token for token in tokens if token not in (self.stoplist)]

        # Lemmatization of tokens for both noun and verb POS and creating a unique list again
        tokens = [self.lmtzr.lemmatize(i) for i in tokens]
        tokens = [self.lmtzr.lemmatize(i,'v') for i in tokens]

        return tokens


    
    def cleanse_tokens(self, tokens):
        """
        This will remove character level noise from tokens
        :param tokens: A string
        :return tokens: A string
        """  
        # Remove '-' if it exists as a separate token or if a token starts or ends with it
        remove_list = ('-','--', 'x')
        tokens = [x for x in tokens if not x in remove_list]

        # Remove single character tokens
        tokens = [x for x in tokens if len(x) >1]

        return tokens


    def preprocess_text(self, sentence):
        """
        This will preprocess a document by cleansing sentences, then lemmatization of tokens and
         by removal of character level noise from tokens
        :param sentence: A string
        :return: sentence
        """
        # Clean the sentence from special characters
        sentence = self.preprocess_sentence(sentence)

        # Create tokens and lematize
        tokens = self.lematize_tokens(sentence)

        # Cleanse tokens
        tokens = self.cleanse_tokens(tokens)

        tokens = list(tokens)
        return ' '.join(tokens)

    def preprocess_alphabetic_sentence(self, sentence):
        """
        This will preprocess a document by doing sentence level alphabetic cleansing
        :param sentence: A string
        :return: sentence
        """
        sentence = self.preprocess_text(sentence)
        return ' '.join([word.lower() for word in sentence.split() if word.isalpha() and word.lower() !='x'])

    def preprocess_data(self):
            """
            This will fit the encoder for all the unique values and introduce unknown value
            :param sentence: A string
            :return: sentence
            """  
            # Create and register a new `tqdm` instance with `pandas`
            tqdm.pandas()

            for column_config in self.preprocess_config:
                # Cleanse sentence only or  cleanse & tokenize 
                column, flag =  list(column_config.values())
                     
                if flag == 0:
                    # Sentence level cleansing
                    self.log.info('Performing sentence cleaning of {}'.format(column))
                    self.data[column+"_processed"] = self.data[column].fillna(self.na_value).apply(self.preprocess_sentence)

                elif flag == 1:
                    # Tokenize and cleanse
                    self.log.info('Performing tokenization and cleaning of {}'.format(column))
                    self.data[column+"_tokens"] = self.data[column].fillna(self.na_value).apply(self.preprocess_text)

                elif flag == 2:
                    # Convert column to timestamp
                    self.data[column] = pd.to_datetime(self.data[column])

                elif flag == 3:
                    # Sentence level alphabetic cleansing
                    self.log.info('Performing alphabetic cleansing {}'.format(column))
                    self.data[column+"_alpha"] = self.data[column].fillna(self.na_value).apply(self.preprocess_alphabetic_sentence)

            return self.data