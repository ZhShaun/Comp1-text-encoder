'''
Put skipthoughs folder in the same parents.
'''

from skip_thoughts import skipthoughts
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from gensim.models import KeyedVectors

import nltk
import numpy as np
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('punkt')

import re

class Encoder():
    def __init__(self, model_name, path=None):
        self.model_name = model_name
        self.path = path

        if self.model_name == 'word2vec' and self.path == None:
            print("Word2vec must explicity specify downloaded" +
            +"pretrained model path!")

        if self.model_name == 'skip':
            self.model = skipthoughts.load_model()
            self.model = skipthoughts.Encoder(self.model)
        elif self.model_name == 'word2vec':
            self.model = KeyedVectors.load_word2vec_format(datapath(path), binary=True)
    
    def __call__(self, X):
        features = None
        if self.model_name == 'skip':
            features = self.model.encode(word_tokenize(X), verbose=False, use_eos=True)
        elif self.model_name == 'word2vec':
            features = self.model[word_tokenize(X)]
        
        return np.mean(features, axis=0)

def tokenizer(text):
    return re.split('\s|(?<!\d)[,.](?!\d)', text)

def tokenizer_stem_nostop(text):
    porter = PorterStemmer()
    return [porter.stem(w) for w in re.split('\s+', text.strip()) \
            if w not in stop and re.match('[a-zA-Z]+', w)]

def tokenizer_stem(text):
    porter = PorterStemmer()
    return [porter.stem(w) for w in re.split('\s+', text.strip())]


# in case of future usage
class model_cfg():
    def __init__(self, name, path=None):
        # name:
        # current options: skip, word2vec
        # pretrained model path .bin if you are using word2vec
        self.name = name
        self.path = path

        if self.name == 'word2vec' and self.path == None:
            print("Word2vec must explicity specify downloaded" +
            +"pretrained model path!")