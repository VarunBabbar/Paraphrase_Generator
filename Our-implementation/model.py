import numpy as np
import torch.nn as nn
import torch
from torch.nn import init


class Model(nn.Module):

    def __init__(self,max_sentence_length,input_size,e_hidden_size,dropout,vocab_size,e_num_layers,batch_size,embedding_dims, d_hidden_size, d_num_layers =1,bidirectional= False):
        super(Model,self).__init__()
        self.input_size = input_size
        self.hidden_size = e_hidden_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.e_num_layers = e_num_layers*2 if bidirectional else e_num_layers
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size,input_size,sparse=False) # Lookup table with word vectors and word index in vocab
        self.lstm_layer = nn.LSTM(input_size, e_hidden_size, e_num_layers, batch_first=False,bidirectional = self.bidirectional,dropout = dropout) # LSTM moodule
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
        self.max_sentence_length = max_sentence_length

        self.embedding_dims = embedding_dims
        self.d_hidden_size = e_hidden_size
        self.d_num_layers = d_num_layers

    def initialise(self):

        ### Initialise decoder
        self.softmax = nn.Softmax(dim = -1)
        # Size of batch = [None, sentence size, embedding dims]  :. size of Wd = [embedding dims, embedding dims] to map to same space
        self.Wd = nn.Linear(in_features = self.embedding_dims, out_features = self.embedding_dims, bias = False)
        self.Wv = nn.Linear(in_features = self.d_hidden_size, out_features = self.embedding_dims, bias = False)
        # LSTM init with size [input_size, hidden_size, num_layers]
        self.lstm = nn.LSTM(self.embedding_dims, self.d_hidden_size, self.d_num_layers)

        ### Initialise encoder
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
    
    def local_loss(self, q_hat, q_real):
        sentence_length = q_hat.shape[0]
        s = 0
        for i in range(q_hat.shape[0]):
            dot = torch.dot(q_hat[i], q_real[i])
            s += dot
        return -(1/sentence_length)*s

    def global_loss(self,encoded_gt,encoded_pred): # Assuming encoded gt and pred are the hidden state of the encoder LSTM at time t
        loss = 0
        for i in range(bs):
            for j in range(bs):
                dot1 = torch.dot(encoded_pred[e_num_layers-1:,i,encoder.max_sentence_length].squeeze(),encoded_gt[:,j,encoder.max_sentence_length].squeeze())
                dot2 = torch.dot(encoded_pred[e_num_layers-1,i,encoder.max_sentence_length].squeeze(),encoded_gt[:,i,encoder.max_sentence_length].squeeze())
                loss += torch.max(0,dot1-dot2+1)
        return loss