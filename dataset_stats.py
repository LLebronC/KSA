import json, os, numpy as np, random
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from collections import Counter
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn import svm
from sklearn.metrics import *
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
s_words = set(stopwords.words('english'))
import matplotlib.pyplot as plt

plt.style.use('seaborn-deep')

from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec

import time

test_split_size = 0.25
MAX_WORDS = 100


def split(base_path):
    if False:  # os.path.exists(os.path.join(base_path, 'dataset.json')):
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_size, stratify=y)
        f = {}

        json.dump(f, open(os.path.join(base_path, 'dataset.json'), 'w'))
    return X_train, X_test, y_train, y_test


def main(X_train, X_test, y_train, y_test):
    dic_words_sar = {}
    dic_words_nsar = {}
    for count_samples in range(len(X_train)):
        sample = X_train[count_samples]
        sample = filter(lambda wor: not wor in s_words, sample.split())

        if y_train[count_samples]:
            dic_to_add = dic_words_sar
        else:
            dic_to_add = dic_words_nsar
        for word in sample:
            if word in dic_to_add.keys():
                dic_to_add[word] += 1
            else:
                dic_to_add[word] = 1
    common_words = list(set(dic_words_sar.keys()).intersection(dic_words_nsar.keys()))
    words_only_sar = list(set(dic_words_sar.keys()) - set(dic_words_nsar.keys()))
    words_only_nsar = list(set(dic_words_nsar.keys()) - set(dic_words_sar.keys()))

    w = 0.5

    # plt.bar(np.arange(len(common_words)),[dic_words_sar[word] for word in common_words], width=w, color='b', align='center',label='The Onion')
    # plt.bar(np.arange(len(common_words))+len(common_words)*[w],[dic_words_nsar[word] for word in common_words], width=w, color='g', align='center',label='HuffPost')
    # plt.legend(loc='upper right')
    # # plt.xlabel(common_words)
    # plt.show()

    k_sar = Counter(dic_words_sar)
    more_use_words_sar = k_sar.most_common(10)
    plt.figure()
    plt.bar(np.arange(len(more_use_words_sar)), [dic_words_sar[word[0]] for word in more_use_words_sar], width=w,
            color='b', align='center', label='The Onion')
    plt.bar(np.arange(len(more_use_words_sar)) + len(more_use_words_sar) * [w],
            [dic_words_nsar[word[0]] if word[0] in dic_words_nsar.keys() else 0 for word in more_use_words_sar],
            width=w, color='g', align='center', label='HuffPost')
    plt.legend(loc='upper right')
    plt.xticks(np.arange(len(more_use_words_sar)), [x[0] for x in more_use_words_sar])
    plt.show()

    k_nsar = Counter(dic_words_nsar)
    more_use_words_nsar = k_nsar.most_common(10)
    plt.figure()
    plt.bar(np.arange(len(more_use_words_nsar)),
            [dic_words_sar[word[0]] if word[0] in dic_words_sar.keys() else 0 for word in more_use_words_nsar], width=w,
            color='b', align='center', label='The Onion')
    plt.bar(np.arange(len(more_use_words_nsar)) + len(more_use_words_nsar) * [w],
            [dic_words_nsar[word[0]] for word in more_use_words_nsar], width=w, color='g', align='center',
            label='HuffPost')
    plt.legend(loc='upper right')
    plt.xticks(np.arange(len(more_use_words_nsar)), [x[0] for x in more_use_words_nsar])
    plt.show()

    return dic_to_add


def embed_words(X):
    data = []

    # iterate through each sentence in the file
    for i in X:
        temp = []

        # tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.lower())

        data.append(temp)

        # Create CBOW model
    model1 = gensim.models.Word2Vec(data, min_count=1,
                                    size=100, window=5)
    return model1


def make_Dictionary(X_train):
    all_words = []
    for line in X_train:
        all_words += line.split()
    all_words = filter(lambda wor: not wor in s_words, all_words)
    dictionary = Counter(all_words)
    dictionary = dictionary.most_common(MAX_WORDS)
    return dictionary


def extract_features(X_train, dictionary):
    features_matrix = np.zeros((len(X_train), MAX_WORDS))
    docID = 0;
    for line in X_train:
        words = line.split()
        for word in words:
            wordID = 0
            for i, d in enumerate(dictionary):
                if d[0] == word:
                    wordID = i
                    features_matrix[docID, wordID] = words.count(word)
        docID = docID + 1
    return features_matrix


# if __name__ == '__main__':
base_path = 'dataset/'
X_train, X_test, y_train, y_test = split(base_path)

out_results=[]
for model in ["MNB","GNB","BNB","SVM"]:
 for MAX_WORDS in [100,300,600,1000,2000,3000]:
    t1=time.time()
    dictionary = make_Dictionary(X_train)
    features_matrix = extract_features(X_train, dictionary)
    if model=="MNB":
        # model=embed_words(X_train)
        clf = MultinomialNB()
    elif model=="SVM":
        clf = svm.SVC(gamma='scale')
    elif model == "GNB":
        clf = GaussianNB()
    elif model == "BNB":
        clf = BernoulliNB()
    clf.fit(features_matrix, y_train)
    # a=main(X_train, X_test, y_train, y_test)

    test_matrix = extract_features(X_test, dictionary)
    result = clf.predict(test_matrix)
    print(accuracy_score(y_test,result))
    t2=time.time()
    out_results.append([model,MAX_WORDS,accuracy_score(y_test,result),t2-t1])