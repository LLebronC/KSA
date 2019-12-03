import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from gensim.models import Word2Vec

from collections import Counter

time_steps = 10
batch_size = 3
in_size = 10
classes_no = 2

from tools import split
import config

from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec

import nltk
from nltk.corpus import stopwords

def make_Dictionary(X_train,Max_Words):
    nltk.download('stopwords')
    s_words = set(stopwords.words('english'))
    all_words = []
    for line in X_train:
        all_words += line.split()
    all_words = filter(lambda wor: not wor in s_words, all_words)
    dictionary = Counter(all_words)
    dictionary = dictionary.most_common(Max_Words)
    return dictionary

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

def extract_features(X_train, dictionary,in_size):
    features_matrix = np.zeros((len(X_train), in_size))
    docID = 0
    for line in X_train:
        words = line.split()
        count_words=0
        pos_word=0
        while(count_words<in_size and pos_word <len(words)):

            word=words[pos_word]
            wordID = 0
            for i, d in enumerate(dictionary):
                if d[0] == word:
                    features_matrix[docID, count_words] = i+1
                    count_words+=1
            pos_word+=1
        docID = docID + 1
    return features_matrix

X, X_test, y, y_test = split(config.base_path)

dictionary = make_Dictionary(X,3000)
features_matrix = extract_features(X, dictionary,in_size)


input_seq = torch.from_numpy(features_matrix)
target_seq = torch.Tensor(y)

device = torch.device("cuda")



model = nn.LSTM(in_size, classes_no, 2)

lr=0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

n_epochs=10
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()  # Clears existing gradients from previous epoch
    input_seq.to(device)
    output, hidden = model(input_seq.float())
    loss = criterion(output, target_seq.view(-1).long())
    loss.backward()  # Does backpropagation and calculates gradients
    optimizer.step()  # Updates the weights accordingly

    print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
    print("Loss: {:.4f}".format(loss.item()))