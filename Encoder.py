#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
from torch.nn import init

# Loading all training examples, validation examples, and vocabulary


class Encoder(nn.Module): # Encoder with a single layer LSTM 
    def __init__(self,input_size,hidden_size,dropout,vocab_size,num_layers, embedded_words, bidirectional= False):
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
        self.embedded_words = embedded_words
        self.init_weights()
        
    def forward(self, x, prev_state, for_decode):  # X corresponds to the index of the word in the vocabulary (so X can be 2 or 49 etc, and embedding(x) will give the word vector of the 2nd or 49th word in the vocabulary)
        embed = self.embedding(x) # Looking up embeddings
        output, curr_state = self.dropout(self.lstm_layer(embed, prev_state))

        # Here curr_state is the tuple (hidden state, cell state)
        return output,curr_state
    
    def init_weights(self): # Initialising weights
        init.orthogonal_(self.lstm_layer.weight_ih_l0)
        init.uniform_(self.lstm_layer.weight_hh_l0, a=-0.01, b=0.01) # Values taken from the paper
        embedding_weights = torch.FloatTensor(self.vocab_size, self.input_size)
        init.uniform_(embedding_weights, a=-0.25, b=0.25) # Values taken from the paper
        for k,v in self.embedded_words.items():
            embedding_weights[k,:] = torch.FloatTensor(v)
        embedding_weights[0] = torch.FloatTensor([0]*self.input_size)
        del self.embedding.weight # Deleting any pre-initialised embeddings in nn.embedding.weights and replacing it with our known word embeddings
        self.embedding.weight = nn.Parameter(embedding_weights)



def trial_encoder():    

    input_size = 100
    hidden_size = 128
    num_layers = 1
    vocab_size = len(vocab)
    dropout = 0.2
    bidirectional = False

    encoder = Encoder(input_size,hidden_size,dropout,vocab_size,num_layers,bidirectional)

    loss_fn = torch.nn.BCELoss()

    learning_rate = 0.001
    num_epochs = 30
    batch_size = 512

    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    def train(model,num_epochs,optimizer,lr): # Have to implement this
        return