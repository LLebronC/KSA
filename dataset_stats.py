import json, os, numpy as np, random

from collections import Counter

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
s_words = set(stopwords.words('english'))
import matplotlib.pyplot as plt

plt.style.use('seaborn-deep')

from tools import split
import config

import matplotlib

font = {'family' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

def main(X_train, X_test, y_train, y_test):
    dic_words_sar = {}
    dic_words_nsar = {}
    list_len_sar=[]
    list_len_nsar = []
    for count_samples in range(len(X_train)):
        sample_r = X_train[count_samples]
        sample = filter(lambda wor: not wor in s_words, sample_r.split())

        if y_train[count_samples]:
            dic_to_add = dic_words_sar
            list_len_sar.append(len(sample_r))
        else:
            dic_to_add = dic_words_nsar
            list_len_nsar.append(len(sample_r))
        for word in sample:
            if word in dic_to_add.keys():
                dic_to_add[word] += 1
            else:
                dic_to_add[word] = 1
    box_plot_data=[list_len_sar,list_len_nsar]
    plt.boxplot(box_plot_data,labels=['sarcastic','no_sarcastic'])
    # fig, axs = plt.subplots(1, 2)
    # axs[0].boxplot(list_len_sar)
    # axs[1].boxplot(list_len_nsar)

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
    more_use_words_sar = k_sar.most_common(20)
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
    more_use_words_nsar = k_nsar.most_common(20)
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

    return [common_words,words_only_sar,words_only_nsar]




if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split(config.base_path)
    a=main(X_train, X_test, y_train, y_test)
