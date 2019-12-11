import time
import json, os, numpy as np, random

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn import svm

from sklearn.metrics import *
from sklearn.model_selection import KFold
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from tools import split, make_Dictionary
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


def extract_features(X_train, dictionary, Max_Words):
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

n_fold = 5
kf = KFold(n_splits=n_fold)
count_folds = 0
out_results = []
for model in ["MNB"]:  # ["MNB", "GNB", "BNB", "SVM"]:
    for Max_Words in [100, 300, 600, 1000, 3000, 20000]:
        metric = 0
        t1 = time.time()
        # for train_index, val_index in kf.split(X):
        # X_train, X_val = [X[index_tx] for index_tx in train_index], [X[index_vx] for index_vx in val_index]
        # y_train, y_val = [y[index_ty] for index_ty in train_index], [y[index_vy] for index_vy in val_index]

        dictionary = make_Dictionary(X, Max_Words)
        features_matrix = extract_features(X, dictionary, Max_Words)
        if model == "MNB":
            param_grid = [
                {'alpha': [0.001, 0.5, 1]},
            ]
            clf = MultinomialNB()
        elif model == "SVM":
            param_grid = [
                {'C': [1, 5, 10], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
            ]
            clf = svm.SVC()

        elif model == "GNB":
            param_grid = [
                {'var_smoothing': [1e-10, 1e-8, 1e-5]},
            ]
            clf = GaussianNB()
        elif model == "BNB":
            param_grid = [
                {'alpha': [0.001, 0.5, 1]},
            ]
            clf = BernoulliNB()
        clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5,scoring=['accuracy', 'precision','recall'],refit='accuracy')
        clf.fit(features_matrix, y)

        pickle.dump(clf, open(os.path.join('models', str(count_folds) + model + str(Max_Words) + '_svm.dmp'), 'wb'))
        # val_matrix = extract_features(X_val, dictionary,Max_Words)
        # result = clf.predict(val_matrix)
        # metric+=accuracy_score(y_val,result)
        # out_results.append([model,Max_Words,metric,sorted(clf.cv_results_.keys())])
        t2 = time.time()
        out_results.append([model, Max_Words, sorted(clf.cv_results_)])
json.dump(out_results, open(os.path.join('models', 'results_svm.json'), 'w'))

best_model = clf
y_score = best_model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(y_test, y_score)
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

confusion_matrix(y_test, y_score)
