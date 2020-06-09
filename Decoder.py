import numpy as np
import torch.nn as nn
import torch

class Decoder(nn.Module):

    def __init__(self, Encoder, embedding_dims, hidden_size, num_layers = 1):
        super(Decoder, self).__init__()

        self.Encoder = Encoder

        ### QUESTION: What dimension for across q?
        self.softmax = nn.Softmax(dim = -1)                 # or 2?

        # Hyper-parameters
        self.embedding_dims = embedding_dims
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.init_weights(embedding_dims, hidden_size, num_layers)

    def local_loss_contrib(self, q_hat, q_real):
        ### QUESTION: In the paper they have, what looks like, average across text, right?
        sentence_length = q_hat.shape[0]

        
        s = 0
        
        for i in range(q_hat.shape[0]):
            print(i)
            
            dot = torch.dot(q_hat[i], q_real[i])
            s += dot
            print(dot)

        return -(1/sentence_length)*s


    def init_weights(self, embedding_dims, hidden_size, num_layers):
        
        # Size of batch = [None, sentence size, embedding dims]  :. size of Wd = [embedding dims, embedding dims] to map to same space
        self.Wd = nn.Linear(in_features = embedding_dims, out_features = embedding_dims, bias = False)
        self.Wv = nn.Linear(in_features = hidden_size, out_features = embedding_dims, bias = False)

        # LSTM init with size [input_size, hidden_size, num_layers]
        self.lstm = nn.LSTM(embedding_dims, hidden_size, num_layers)        
    
    def forward(self, sentence, q_real):
        """
        A forward pass on the LSTM decoder. Input and ground truth come in size [sentence length, batch size, embedding dims]
        """

        if self.Encoder == None:
            # Initialise hidden layer and cell state, size = [num_layers*num_directions, batch, hidden_size]
            h0 = torch.zeros((self.num_layers, sentence.size()[1], self.hidden_size))
            c0 = torch.zeros((self.num_layers, sentence.size()[1], self.hidden_size))
        else:
            pass

        # dt = Wd * qt = d-weights * previously predicted word
        dt = self.Wd(sentence)

        # h[t+1] = LSTM(dt, ht)          LSTM output paramter = [sentence length, batch, hidden_size*num_directions]
        out, (hn, cn) = self.lstm(dt, (h0, c0))
        
        # Out has dims: [sentence length, batch, embedding_dims]    
        p = self.Wv(out)

        # Softmax p to get q_hat
        q_hat = self.softmax(p)
        q_real = torch.mean(q_real, axis = 1)
        q_hat = torch.mean(q_hat, axis = 1)

        # CROSS ENTROPY TAKEN OVER dim 0 (i.e. sentence length)
        # This is one of the terms at in the L_local summation, which must later be added up and divided
        local_loss = self.local_loss_contrib(q_hat, q_real) /sentence.size()[1]

        return local_loss, q_hat

    def train(self, num_epochs, num_iterations, learning_rate):
        self.optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for i in range(num_iterations):
                pass ### waiting for encoder output


def trial_forward_run():
    embedding_dims = 100
    hidden_size = 20

    decoder = Decoder(embedding_dims, hidden_size)

    inp = torch.randn(5, 3, embedding_dims)
    gtt = torch.randn(5, 3, embedding_dims)

    loss, q_hat = decoder.forward(inp, gtt)
    print(loss)