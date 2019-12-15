import time
import  os, numpy as np

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn import svm

from sklearn.metrics import *

import matplotlib.pyplot as plt
from tools import split, make_Dictionary
import config

import pickle


from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sn
#word frequency feature
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


if __name__ == '__main__':
    feature=config.features
    X, X_test, y, y_test = split(config.base_path)

    out_results = []
    for model in ["MNB", "GNB", "BNB", "SVM"]:
        #take care of SVM apart to not try so many vocaublary sizes
        if model != "SVM":
            #maximum number of diferent words
            for Max_Words in [100, 300, 600, 1000, 3000, 20000]:

                t1 = time.time()
                #select feature and recomputed the dictionary
                if feature=="tdidf":
                    vectorizer = TfidfVectorizer(stop_words='english',lowercase=True,max_features=Max_Words)
                    features_matrix = vectorizer.fit_transform(X)
                    features_matrix = features_matrix.toarray()
                else:
                    dictionary = make_Dictionary(X, Max_Words)
                    features_matrix = extract_features(X, dictionary, Max_Words)

                #each model with their hyoperparameter optimitzaion
                if model == "MNB":
                    param_grid = [
                        {'alpha': [0.001, 0.5, 1]},
                    ]
                    clf = MultinomialNB()
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
                #grid search of the hyperparameter with crossvalidation 5-folds
                clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5,scoring=['accuracy', 'precision','recall','f1','balanced_accuracy'],refit='accuracy')
                clf.fit(features_matrix., y)
                #save the model to disk
                pickle.dump(clf, open(os.path.join('models', str(count_folds) + model + str(Max_Words) + '.dmp'), 'wb'))

                t2 = time.time()
                #save the results for later
                out_results.append([model, Max_Words, clf.cv_results_,os.path.join('models', str(count_folds) + model + str(Max_Words) + '.dmp')])
        #same but for the SVM
        if model == "SVM":
            for Max_Words in [100, 300]:

                t1 = time.time()

                if feature=="tdidf":
                    vectorizer = TfidfVectorizer(stop_words='english',lowercase=True,max_features=Max_Words)
                    features_matrix = vectorizer.fit_transform(X)
                    features_matrix = features_matrix.toarray()
                else:
                    dictionary = make_Dictionary(X, Max_Words)
                    features_matrix = extract_features(X, dictionary, Max_Words)


                param_grid = [
                    {'C': [1, 5, 10], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
                ]
                clf = svm.SVC(probability=True)
                clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5,scoring=['accuracy', 'precision','recall','f1','balanced_accuracy'],refit='accuracy')
                clf.fit(features_matrix, y)

                pickle.dump(clf, open(os.path.join('models', str(count_folds) + model + str(Max_Words) + '.dmp'), 'wb'))

                t2 = time.time()
                out_results.append([model, Max_Words, clf.cv_results_,os.path.join('models', str(count_folds) + model + str(Max_Words) + '.dmp')])
        #dump the results to recomputed everything on an error
        pickle.dump(out_results, open(os.path.join('models', 'results.pk'), 'wb'))
    # find the model with the max accuracy (already the default in the gridsearch object)
    max_accuracy_model=np.argmax([max(x[2]['mean_test_accuracy']) for x in out_results])
    best_model = pickle.load(open(out_results[max_accuracy_model][3],'rb'))
    #recomputed the dictionary,would has been better to save them but we will use only one at the end
    if feature == "tdidf":
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True,max_features=out_results[max_accuracy_model][1])
        features_matrix = vectorizer.fit_transform(X)
        features_test = vectorizer.transform(X_test)
        features_matrix = features_matrix.toarray()

    else:
        dictionary = make_Dictionary(X, out_results[max_accuracy_model][1])
        features_test = extract_features(X_test, dictionary, out_results[max_accuracy_model][1])
    #predict in test
    y_score = best_model.predict_proba(features_test)
    #some metrics
    metrics={
        'accuracy_score':accuracy_score(y_test,np.argmax(y_score,axis=1)),
        'balanced_accuracy_score': balanced_accuracy_score(y_test, np.argmax(y_score, axis=1)),
        'precision_recall_fscore_support': precision_recall_fscore_support(y_test, np.argmax(y_score, axis=1),average='micro')}
    print(metrics)
    #computed and plot roc curve with area under the curve
    fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    #computed and plot the confusion matrix
    cm=confusion_matrix(y_test,np.argmax(y_score,axis=1))
    sn.heatmap(cm, xticklabels=['no sarcastic','sarcastic'],yticklabels=['no sarcastic','sarcastic'], cmap="YlGnBu")
