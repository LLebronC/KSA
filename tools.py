import json, os, numpy as np, random
import config

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import nltk
from nltk.corpus import stopwords

from collections import Counter

nltk.download('stopwords')
s_words = set(stopwords.words('english'))
#creat the dictionari of the Max_Words most used words removing common words
def make_Dictionary(X_train,Max_Words):

    all_words = []
    for line in X_train:
        all_words += line.split()
    all_words = filter(lambda wor: not wor in s_words, all_words)
    dictionary = Counter(all_words)
    dictionary = dictionary.most_common(Max_Words)
    return dictionary



# create or load train and test split
def split(base_path):
    if os.path.exists(os.path.join(base_path, 'dataset.json')):
        f = json.load(open(os.path.join(base_path, 'dataset.json'), 'r'))
        X_train = f['X_train']
        X_test = f['X_test']
        y_train = f['y_train']
        y_test = f['y_test']
    else:

        sarcasm_file = json.load(open(os.path.join(base_path, 'fake_news.json'), 'r'))
        headlines = [[news['headline'], int(news['is_sarcastic'])] for news in sarcasm_file]
        X = [news[0] for news in headlines]
        y = [news[1] for news in headlines]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_split_size, stratify=y)
        f = {
            'X_train':X_train,
            'X_test':X_test,
            'y_train':y_train,
            'y_test':y_test,

        }

        json.dump(f, open(os.path.join(base_path, 'dataset.json'), 'w'))
    return X_train, X_test, y_train, y_test
