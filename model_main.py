import time
import json, os, numpy as np, random

from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn import svm

from sklearn.metrics import *
from sklearn.model_selection import KFold
from collections import Counter

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
s_words = set(stopwords.words('english'))

from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec

from tools import split
import config

import pickle

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


def make_Dictionary(X_train,Max_Words):
    all_words = []
    for line in X_train:
        all_words += line.split()
    all_words = filter(lambda wor: not wor in s_words, all_words)
    dictionary = Counter(all_words)
    dictionary = dictionary.most_common(Max_Words)
    return dictionary


def extract_features(X_train, dictionary,Max_Words):
    features_matrix = np.zeros((len(X_train), Max_Words))
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

X, X_test, y, y_test = split(config.base_path)


kf = KFold(n_splits=5)
count_folds=0
out_results=[]
for train_index, val_index in kf.split(X):
    X_train, X_val = [X[index_tx] for index_tx in train_index], [X[index_vx] for index_vx in val_index]
    y_train, y_val = [y[index_ty] for index_ty in train_index], [y[index_vy] for index_vy in val_index]

    for model in ["MNB","GNB","BNB","SVM"]:
     for Max_Words in [100,300,600,1000,2000,3000,20000]:
        t1=time.time()
        dictionary = make_Dictionary(X_train,Max_Words)
        features_matrix = extract_features(X_train, dictionary,Max_Words)
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

        pickle.dump(clf,open(os.path.join('models',str(count_folds)+model+str(Max_Words)+'.dmp'),'wb'))
        val_matrix = extract_features(X_val, dictionary,Max_Words)
        result = clf.predict(val_matrix)
        print(accuracy_score(y_val,result))
        t2=time.time()
        out_results.append([count_folds,model,Max_Words,accuracy_score(y_val,result),t2-t1])
        count_folds+=1
json.dump(out_results,open(os.path.join('models','results.json'),'w'))