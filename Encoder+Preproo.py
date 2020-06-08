#!/usr/bin/env python
# coding: utf-8

# In[248]:


import fastai
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import json 
import os
import pickle

# Loading all training examples, validation examples, and vocabulary

with open('/Users/varunbabbar/Downloads/Data_Quora/quora_raw_train.json') as f:
    training_examples = json.load(f)
    
with open('/Users/varunbabbar/Downloads/Data_Quora/quora_raw_val.json') as f:
    val_examples = json.load(f)

with open('/Users/varunbabbar/Downloads/Data_Quora/quora_data_prepro.json') as f:
    vocab = json.load(f)['ix_to_word'] # Vocab = {1: 'the', 2: 'sauce' .....}
vocab_inv = {word:number for number,word in list(vocab.items())} # Inverted dictionary created          Vocab_inv = {'the':1, 'sauce':2, .....}

X_train_prelim = [training_examples[i]['question1'] for i in range(len(training_examples))] # Training examples before preprocessing
Y_train_prelim = [training_examples[i]['question'] for i in range(len(training_examples))]

X_val_prelim = [val_examples[i]['question1'] for i in range(len(val_examples))]
Y_val_prelim = [val_examples[i]['question'] for i in range(len(val_examples))]

X_train = []
Y_train = []

def load_glove_embeddings(filename="glove.6B.100d.txt"):
    lines = open(filename).readlines()
    print(len(lines))
    embeddings = {}
    a = 0
    for line in lines:
        a +=1
        print(a)
        word = line.split()[0]
        embedding = list(map(float, line.split()[1:]))
        if word in vocab.values():
            embeddings[int(vocab_inv[word])] = embedding # This is creating a dictionary with indexes corresponding to the position of the particular word in the vocabulary
            # embeddings = {1:embedding of 'the', 2: embedding of 'sauce'....}
    return embeddings

embedded_words = load_glove_embeddings()

for i in range(len(X_train_prelim)): # This is basically tokenising each sentence, and converting it into a list of word embeddings. X corresponds to a training example and Y corresponds to its paraphrase
    print(i)
    sentence_x = X_train_prelim[i]
    sentence_y = Y_train_prelim[i]
    tokensx = word_tokenize(sentence_x)
    tokensy = word_tokenize(sentence_y)
    vectorx = []
    vectory = []
    for word in tokensx:
        try:
            vectorx.append(embedded_words[int(vocab_inv[word])])
        except:
            vectorx.append(-2*torch.rand(100,1)) # If the word is not in the embeddings given, then add a random vector (might have to do something about this :) ) (Potentially could write a spell checker to find misspellings)
    for word in tokensy:
        try:
            vectorx.append(embedded_words[int(vocab_inv[word])])
        except:
            vectory.append(-2*torch.rand(100,1))

    X_train.append(vectorx)
    Y_train.append(vectory)


input_size = 100
hidden_size = 128
num_layers = 1
vocab_size = len(vocab)
dropout = 0.2
bidirectional = False

class Encoder(nn.Module): # Encoder with a single layer LSTM 
    def __init__(self,input_size,hidden_size,dropout,vocab_size,num_layers,bidirectional= False):
        super(Encoder,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size,input_size,sparse=False) # Lookup table with word vectors and word index in vocab
        self.lstm_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,bidirectional = False) # LSTM moodule
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
        
    def forward(self, x, prev_state):  # X corresponds to the index of the word in the vocabulary (so X can be 2 or 49 etc, and embedding(x) will give the word vector of the 2nd or 49th word in the vocabulary)
        embed = self.embedding(x) # Looking up embeddings
        output, curr_state = self.dropout(self.lstm_layer(embed, prev_state))
        return output,curr_state
    
    def init_weights(self): # Initialising weights
        init.orthogonal_(self.lstm_layer.weight_ih_l0)
        init.uniform_(self.lstm_layer.weight_hh_l0, a=-0.01, b=0.01) # Values taken from the paper
        embedding_weights = torch.FloatTensor(self.vocab_size, self.input_size)
        init.uniform_(embedding_weights, a=-0.25, b=0.25) # Values taken from the paper
        for k,v in embedded_words.items():
            embedding_weights[k,:] = torch.FloatTensor(v)
        embedding_weights[0] = torch.FloatTensor([0]*self.input_size)
        del self.embedding.weight # Deleting any pre-initialised embeddings in nn.embedding.weights and replacing it with our known word embeddings
        self.embedding.weight = nn.Parameter(embedding_weights)
    
encoder = Encoder(input_size,hidden_size,dropout,vocab_size,num_layers,bidirectional)

loss_fn = torch.nn.BCELoss()

learning_rate = 0.001
num_epochs = 30
batch_size = 512

optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

def train(model,num_epochs,optimizer,lr): # Have to implement this
    return


# In[244]:


print(np.dot(w2v['rock'],w2v['banana']))






