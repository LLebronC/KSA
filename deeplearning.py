import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from gensim.models import Word2Vec

from collections import Counter

from tools import split
import config
import time
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

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out

in_size = 10

X, X_test, y, y_test = split(config.base_path)

dictionary = make_Dictionary(X,3000)
features_matrix = extract_features(X, dictionary,in_size)


features_matrix_val = extract_features(X_test, dictionary,in_size)


input_seq = torch.from_numpy(np.expand_dims(features_matrix, axis=1))
target_seq = torch.Tensor(np.array(y)).long()


input_seq_val=torch.from_numpy(np.expand_dims(features_matrix_val, axis=1))
target_seq_val= torch.Tensor(np.array(y_test)).long()

device = torch.device("cuda")



model = LSTMModel(in_size,100,1,2)

lr=0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)



total = target_seq.size(0)
total_val = target_seq_val.size(0)
n_epochs=1000
t1=time.time()
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()  # Clears existing gradients from previous epoch
    input_seq.to(device)
    output = model(input_seq.float())
    loss = criterion(output, target_seq)
    loss.backward()  # Does backpropagation and calculates gradients
    optimizer.step()  # Updates the weights accordingly

    _,predicted = torch.max(output.data, 1)
    tp=(predicted == target_seq).sum()


    output_val = model(input_seq_val.float())
    _,predicted_val = torch.max(output_val.data, 1)
    tp_val=(predicted_val == target_seq_val).sum()

    print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
    print("Loss: {:.4f}".format(loss.item()))
    print("Accuracy_train: {}".format(float(tp)/total))
    print("Accuracy_val: {}".format(float(tp_val)/total_val))
print(time.time()-t1)