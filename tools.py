import json, os, numpy as np, random
import config

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


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

