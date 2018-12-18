import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

def tokenize(t):
    """ Customized tokenizer called by TfidfVectorizer.
        Tokenize tweets using TweetTokenizer and lemmatize
        each token with WordNetLemmatizer. """
    tweet_tok = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = tweet_tok.tokenize(t)
    wnl = WordNetLemmatizer()
    stems = []
    for item in tokens:
        stems.append(wnl.lemmatize(item))
    return stems

def import_data(full=False):
    """ Import the test tweets, the train tweets positive and the train
        tweets negative. Import small sets of full sets depending if the
        'full' parameter is set to True of False.
        Return pandas dataframe for each of the 3 sets. """
    if(full):
        tweet_pos = pd.read_csv('data/train_pos_full.txt', header = None, sep = "\r\n", engine = 'python')
        tweet_neg = pd.read_csv('data/train_neg_full.txt', header = None, sep = "\r\n", engine = 'python')
    else:
        tweet_pos = pd.read_csv('data/train_pos.txt', header = None, sep = "\r\n", engine = 'python')
        tweet_neg = pd.read_csv('data/train_neg.txt', header = None, sep = "\r\n", engine = 'python')
    tweet_test = pd.read_csv('data/test_data.txt', header = None, sep = "\r\n", engine = 'python')
    return tweet_pos, tweet_neg, tweet_test

def clean_data(array):
    """ Clean the data by deleting the id
        placed in the front of the tweet. """
    ret = np.zeros(len(array))
    for i in range(len(array)):
        drop_id = len(str(i+1)) + 1
        array[i, 0] = array[i, 0][int(drop_id):]
    return array

def zero_to_neg(array):
    """ Given an array of 0 and 1, transform it into
    an array of -1 and 1. """
    ret = np.ones(len(array))
    for i in range(len(array)):
        if(array[i] == 0):
            ret[i] = -1
    return ret

def build_submission(y_pred, id_submission):
    """ Build submission and save it into the
        folder 'prediction' with id 'id_submission'."""
    y_pred_ = zero_to_neg(y_pred)
    ret = np.ones((len(y_pred_), 2))
    for i in range(len(y_pred_)):
        ret[i] = np.array([i+1, y_pred_[i]])
    ret = ret.astype(int)
    sub = pd.DataFrame(data = ret)
    sub.columns = ['Id', 'Prediction']
    sub.to_csv('predictions/pred' + id_submission + '.csv', index=None)