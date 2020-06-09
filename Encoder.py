#!/usr/bin/env python
# coding: utf-8

# In[82]:


#!/usr/bin/env python
# coding: utf-8

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
import requests
from torch.nn.utils.rnn import pad_sequence
# # Loading all training examples, validation examples, and vocabulary

with open('quora_raw_train.json') as f:
    training_examples = json.load(f)
    
with open('quora_raw_val.json') as f:
    val_examples = json.load(f)

with open('quora_data_prepro.json') as f:
    vocab = json.load(f)['ix_to_word'] # Vocab = {1: 'the', 2: 'sauce' .....}
vocab_inv = {word:number for number,word in list(vocab.items())} # Inverted dictionary created   Vocab_inv = {'the':1, 'sauce':2, .....}

X_train_prelim = [training_examples[i]['question1'] for i in range(len(training_examples))] # Training examples before preprocessing
Y_train_prelim = [training_examples[i]['question'] for i in range(len(training_examples))]

X_val_prelim = [val_examples[i]['question1'] for i in range(len(val_examples))]
Y_val_prelim = [val_examples[i]['question'] for i in range(len(val_examples))]

def load_glove_embeddings(filename="glove.6B.100d.txt"):

    with open("embeddings", "rb") as f:
        embeddings = pickle.load(f)
        return embeddings

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

def tokenize_embed(X,Y): # Doing this for training and validation_datasets
    X_train = []
    Y_train = []
    for i in range(len(X)): # This is basically tokenising each sentence, and converting it into a list of word embeddings. X corresponds to a training example and Y corresponds to its paraphrase
        print(i)
        sentence_x = X[i]
        sentence_y = Y[i]
        tokensx = word_tokenize(sentence_x.lower())
        tokensy = word_tokenize(sentence_y.lower())
        vectorx = []
        vectory = []
        for word in tokensx:
            try:
                vectorx.append(int(vocab_inv[word]))
            except:
                vectorx.append(258) # If the word is not in the embeddings given, then add a ',' instead. (Need to discuss this)
        for word in tokensy:
            try:
                vectory.append(int(vocab_inv[word]))
            except:
                vectory.append(258)
        X_train.append(vectorx)
        Y_train.append(vectory)
    return X_train,Y_train

def clip_pad_input_sequences(X,Y,min_sentence_length,max_sentence_length): # Doing this for training and validation datasets
    X_training_clipped = []
    Y_training_clipped = []
    for i in range(len(X)):
        print(i,X[i])
        if len(X[i]) >= min_sentence_length and len(X[i]) <= max_sentence_length and len(Y[i]) >= min_sentence_length and len(Y[i]) <= max_sentence_length:
            X_training_clipped.append(torch.Tensor(X[i]))
            Y_training_clipped.append(torch.Tensor(Y[i]))
    padded_X_training_clipped = pad_sequence(X_training_clipped,padding_value = vocab_size)
    padded_Y_training_clipped = pad_sequence(X_training_clipped,padding_value = vocab_size)
    return padded_X_training_clipped,padded_Y_training_clipped

def find_closest_embedding(output,vocab,embedded_words): # Need this because we probably need to find the closest embedding to a given embedding and thereby find the word
    return

min_sentence_length = 7
min_sentence_length = 16

embedded_words = load_glove_embeddings()



input_size = 100 # Size of embedding
hidden_size = 256 # Can set this to be a custom value
num_layers = 1 # Can change this
bs = 128
vocab_size = len(vocab)
dropout = 0.2
bidirectional = False
max_sentence_length = 16 # Max number of words in a sentence


X_train,Y_train = clip_pad_input_sequences(*tokenize_embed(X_train_prelim,Y_train_prelim,),min_sentence_length,max_sentence_length)
X_valid,Y_valid = clip_pad_input_sequences(*tokenize_embed(X_val_prelim,Y_val_prelim,),min_sentence_length,max_sentence_length)


class Encoder(nn.Module): # Encoder with a single layer LSTM 
    def __init__(self,max_sentence_length,input_size,hidden_size,dropout,vocab_size,num_layers,bs,bidirectional= False):
        super(Encoder,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bs = bs
        self.num_layers = num_layers*2 if bidirectional else num_layers
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size,input_size,sparse=False) # Lookup table with word vectors and word index in vocab
        self.lstm_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False,bidirectional = self.bidirectional,dropout = dropout) # LSTM moodule
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
        self.max_sentence_length = max_sentence_length
        
    def forward(self, x, prev_state):  # X corresponds to the index of the word in the vocabulary (so X can be 2 or 49 etc, and embedding(x) will give the word vector of the 2nd or 49th word in the vocabulary)
        embed = self.embedding(x) # Looking up embeddings
        embed = torch.reshape(torch.Tensor(embed),(self.max_sentence_length,self.bs,self.input_size))
        output, curr_state = self.lstm_layer(embed, prev_state)
        return output,curr_state, embed
    
    def init_weights(self): # Initialising weights
        init.orthogonal_(self.lstm_layer.weight_ih_l0)
        init.uniform_(self.lstm_layer.weight_hh_l0, a=-0.01, b=0.01) # Values taken from the paper
        embedding_weights = torch.FloatTensor(self.vocab_size+1, self.input_size)
        init.uniform_(embedding_weights, a=-0.25, b=0.25) # Values taken from the paper
        for k,v in embedded_words.items():
            embedding_weights[k,:] = torch.FloatTensor(v)
        embedding_weights[0] = torch.FloatTensor([0]*self.input_size)
        embedding_weights[vocab_size] = torch.FloatTensor([0]*self.input_size) # Equivalent embedding for 
        del self.embedding.weight # Deleting any pre-initialised embeddings in nn.embedding.weights and replacing it with our known word embeddings
        self.embedding.weight = nn.Parameter(embedding_weights)
    
    def train(self,x): # Batch feedforward across all training examples
        curr_state = (torch.zeros(self.num_layers,self.bs,self.hidden_size),torch.zeros(self.num_layers,self.bs,self.hidden_size)) # Initialization of hidden and cell state
        for i in range(max(x.shape)//bs):
            if i < max(x.shape)//bs-1:
                wordx = x[:,i:i+bs]
                output, curr_state = encoder.forward(wordx.long(),curr_state)
            print(i)
    
def Discriminator():
    def __init__(self,ground_truth_sentence,predicted_sentence,encoder):
        super(Discriminator,self).__init__()
        self.ground_truth_sentence = ground_truth_sentence # Batch input of GT sequence: eg input = [3,444,5234,64,88] where 3 = embedding of word 3 in vocab and so on
        self.predicted_sentence = predicted_sentence # Batch input of predicted sequence: eg input = [3,444,5234,64,88] where 3 = embedding of word 3 in vocab and so on
        self.encoder = encoder # encoder module

    def forward(self): # Prev state is the hidden state and cell state of the encoder at the given timestep
        _,(encoded_gt,_)= encoder.forward(self.ground_truth_sentence,prev_state)
        _,(encoded_pred,_) = encoder.forward(self.predicted_sentence,prev_state)
        loss = self.global_loss(encoded_gt,encoded_pred)
        return loss
        
    def backward(self): # Have to call loss.backward()
        return 0
    
    def global_loss(self,encoded_gt,encoded_pred): # Have to implement global loss_function
        return 0
        
encoder = Encoder(max_sentence_length,input_size,hidden_size,dropout,vocab_size,num_layers,bs,bidirectional)
encoder.train(X_train)
loss_fn = torch.nn.BCELoss()

learning_rate = 0.001
num_epochs = 30
batch_size = 512

optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
# Input = seq length, batch, input_size
# Hidden state = num_layers * num_directions, batch, hidden_size
# Zero-pad the sentences to the same 

