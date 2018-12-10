import numpy as np
import pandas as pd
import re

def tweet_to_vect(tweet, voc_cut, all_vects):
    """ Map a tweet to a vector using the vector of each
        word that is present in the tweet"""
    word_array = re.findall(r'\w+', tweet)
    vect = np.zeros(20)
    for word in word_array:
        i = np.argwhere(voc_cut == word)
        if(i.shape[0] != 0): 
            vect += all_vects[i[0][0]]
    return vect/len(word_array)

def weighted_tweet_to_vect(tweet, voc_cut, all_vects, score_pos, score_neg, dict_):
    """ Map a tweet to a vector using the vector of each
        word that is present in the tweet and multiply it
        by its TF-IDF value. """
    word_array = re.findall(r'\w+', tweet)
    vect = np.zeros(20)
    for word in word_array:
        i = np.argwhere(voc_cut == word)
        if(word in dict_):
            id_in_dict = dict_[word]
            score = np.maximum(score_pos[0, id_in_dict], score_neg[0, id_in_dict])
            if(i.shape[0] != 0): 
                vect += score * all_vects[i[0][0]]
        else:
            if(i.shape[0] != 0): 
                vect += all_vects[i[0][0]]
    return vect/len(word_array)

def clean_data(array):
    """ Clean the data by deleting the id
        placed in the front of the tweet."""
    ret = np.zeros(len(array))
    for i in range(len(array)):
        drop_id = len(str(i+1)) + 1
        array[i, 0] = array[i, 0][int(drop_id):]
    return array

def zero_to_neg(array):
    ret = np.ones(len(array))
    for i in range(len(array)):
        if(array[i] == 0):
            ret[i] = -1
    return ret

def build_submission(y_pred, id_submission):
    """ Build submission and save it into the
        folder 'prediction'."""
    y_pred_ = zero_to_neg(y_pred)
    ret = np.ones((len(y_pred_), 2))
    for i in range(len(y_pred_)):
        ret[i] = np.array([i+1, y_pred_[i]])
    ret = ret.astype(int)
    sub = pd.DataFrame(data = ret)
    sub.columns = ['Id', 'Prediction']
    sub.to_csv('predictions/pred' + id_submission + '.csv', index=None)
