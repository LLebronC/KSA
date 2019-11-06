import json, os, numpy as np, random
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from collections import Counter

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
s_words=set(stopwords.words('english'))
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')


test_split_size = 0.25


def split(base_path):
    if os.path.exists(os.path.join(base_path, 'dataset.json')):
        f=json.load(open(os.path.join(base_path, 'dataset.json'), 'r'))
        X_train= f['X_train']
        X_test = f['X_test']
        y_train = f['y_train']
        y_test = f['y_test']
    else:

        sarcasm_file = json.load(open(os.path.join(base_path, 'fake_news.json'), 'r'))
        headlines = [[news['headline'], int(news['is_sarcastic'])] for news in sarcasm_file]
        X = [news[0] for news in headlines]
        y = [news[1] for news in headlines]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_size, stratify=y)
        f = {}

        json.dump(f,open(os.path.join(base_path, 'dataset.json'), 'w'))
    return X_train, X_test, y_train, y_test


def main(X_train, X_test, y_train, y_test):
    dic_words_sar = {}
    dic_words_nsar = {}
    for count_samples in range(len(X_train)):
        sample=X_train[count_samples]
        sample = filter(lambda wor: not wor in s_words, sample.split())

        if y_train[count_samples]:
            dic_to_add=dic_words_sar
        else:
            dic_to_add = dic_words_nsar
        for word in sample:
            if word in dic_to_add.keys():
                dic_to_add[word] +=1
            else:
                dic_to_add[word] = 1
    common_words=list(set(dic_words_sar.keys()).intersection(dic_words_nsar.keys()))
    words_only_sar = list(set(dic_words_sar.keys()) - set(dic_words_nsar.keys()))
    words_only_nsar = list(set(dic_words_nsar.keys()) - set(dic_words_sar.keys()))

    w=0.5


    # plt.bar(np.arange(len(common_words)),[dic_words_sar[word] for word in common_words], width=w, color='b', align='center',label='The Onion')
    # plt.bar(np.arange(len(common_words))+len(common_words)*[w],[dic_words_nsar[word] for word in common_words], width=w, color='g', align='center',label='HuffPost')
    # plt.legend(loc='upper right')
    # # plt.xlabel(common_words)
    # plt.show()

    k_sar = Counter(dic_words_sar)
    more_use_words_sar = k_sar.most_common(10)
    plt.figure()
    plt.bar(np.arange(len(more_use_words_sar)),[dic_words_sar[word[0]] for word in more_use_words_sar], width=w, color='b', align='center',label='The Onion')
    plt.bar(np.arange(len(more_use_words_sar))+len(more_use_words_sar)*[w],[dic_words_nsar[word[0]] if word[0] in dic_words_nsar.keys() else 0 for word in more_use_words_sar ], width=w, color='g', align='center',label='HuffPost')
    plt.legend(loc='upper right')
    plt.xticks(np.arange(len(more_use_words_sar)),[x[0] for x in more_use_words_sar])
    plt.show()


    k_nsar = Counter(dic_words_nsar)
    more_use_words_nsar = k_nsar.most_common(10)
    plt.figure()
    plt.bar(np.arange(len(more_use_words_nsar)),[dic_words_sar[word[0]] if word[0] in dic_words_sar.keys() else 0 for word in more_use_words_nsar], width=w, color='b', align='center',label='The Onion')
    plt.bar(np.arange(len(more_use_words_nsar))+len(more_use_words_nsar)*[w],[dic_words_nsar[word[0]] for word in more_use_words_nsar], width=w, color='g', align='center',label='HuffPost')
    plt.legend(loc='upper right')
    plt.xticks(np.arange(len(more_use_words_nsar)),[x[0] for x in more_use_words_nsar])
    plt.show()
    return dic_to_add



# if __name__ == '__main__':
base_path = 'dataset/'
X_train, X_test, y_train, y_test=split(base_path)
a=main(X_train, X_test, y_train, y_test)
