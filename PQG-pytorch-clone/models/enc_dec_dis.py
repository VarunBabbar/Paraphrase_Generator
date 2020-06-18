import torch
import torch.nn as nn
import numpy as np
import os
import sys

sys.path.append(".")

import misc.utils as utils

from glob import glob
from misc import net_utils, utils
import re
import pickle
import spacy
from scipy import spatial


class ParaphraseGenerator(nn.Module):
    """
    pytorch module which generates paraphrase of given phrase
    """
    def __init__(self, op):

        super(ParaphraseGenerator, self).__init__()

        # encoder :
        self.emb_layer = nn.Sequential(
            nn.Linear(op["vocab_sz"], op["emb_hid_dim"]),
            nn.Threshold(0.000001, 0),
            nn.Linear(op["emb_hid_dim"], op["emb_dim"]),
            nn.Threshold(0.000001, 0))
        self.enc_rnn = nn.GRU(op["emb_dim"], op["enc_rnn_dim"])
        self.enc_lin = nn.Sequential(
            nn.Dropout(op["enc_dropout"]),
            nn.Linear(op["enc_rnn_dim"], op["enc_dim"]))
        
        # generator :
        self.gen_emb = nn.Embedding(op["vocab_sz"], op["emb_dim"])
        self.gen_rnn = nn.LSTM(op["enc_dim"], op["gen_rnn_dim"])
        self.gen_lin = nn.Sequential(
            nn.Dropout(op["gen_dropout"]),
            nn.Linear(op["gen_rnn_dim"], op["vocab_sz"]),
            nn.LogSoftmax(dim=-1))
        
        # pair-wise discriminator :
        self.dis_emb_layer = nn.Sequential(
            nn.Linear(op["vocab_sz"], op["emb_hid_dim"]),
            nn.Threshold(0.000001, 0),
            nn.Linear(op["emb_hid_dim"], op["emb_dim"]),
            nn.Threshold(0.000001, 0),
        )
        self.dis_rnn = nn.GRU(op["emb_dim"], op["enc_rnn_dim"])
        self.dis_lin = nn.Sequential(
            nn.Dropout(op["enc_dropout"]),
            nn.Linear(op["enc_rnn_dim"], op["enc_dim"]))
        
        # some useful constants :
        self.max_seq_len = op["max_seq_len"]
        self.vocab_sz = op["vocab_sz"]

        self.generateInfo()


    def forward(self, phrase, sim_phrase=None, train=False):
        """
        forward pass

        inputs :-

        phrase : given phrase , shape = (max sequence length, batch size)
        sim_phrase : (if train == True), shape = (max seq length, batch sz)
        train : if true teacher forcing is used to train the module

        outputs :-

        out : generated paraphrase, shape = (max sequence length, batch size, )
        enc_out : encoded generated paraphrase, shape=(batch size, enc_dim)
        enc_sim_phrase : encoded sim_phrase, shape=(batch size, enc_dim)

        """

        if sim_phrase is None:
            sim_phrase = phrase

        if train:

            # encode input phrase
            enc_phrase = self.enc_lin(
                self.enc_rnn(
                    self.emb_layer(utils.one_hot(phrase, self.vocab_sz)))[1])
            
            # generate similar phrase using teacher forcing
            emb_sim_phrase_gen = self.gen_emb(sim_phrase)
            out_rnn, _ = self.gen_rnn(
                torch.cat([enc_phrase, emb_sim_phrase_gen[:-1, :]], dim=0))
            out = self.gen_lin(out_rnn)

            # propagated from shared discriminator to calculate
            # pair-wise discriminator loss
            enc_sim_phrase = self.dis_lin(
                self.dis_rnn(
                    self.dis_emb_layer(utils.one_hot(sim_phrase,
                                                     self.vocab_sz)))[1])
            enc_out = self.dis_lin(
                self.dis_rnn(self.dis_emb_layer(torch.exp(out)))[1])

        else:

            # encode input phrase
            enc_phrase = self.enc_lin(
                self.enc_rnn(
                    self.emb_layer(utils.one_hot(phrase, self.vocab_sz)))[1])
            
            # generate similar phrase using teacher forcing
            words = []
            h = None
            for __ in range(self.max_seq_len):
                word, h = self.gen_rnn(enc_phrase, hx=h)
                word = self.gen_lin(word)
                words.append(word)
                word = torch.multinomial(torch.exp(word[0]), 1)
                word = word.t()
                enc_phrase = self.gen_emb(word)
            out = torch.cat(words, dim=0)

            # propagated from shared discriminator to calculate
            # pair-wise discriminator loss
            enc_sim_phrase = self.dis_lin(
                self.dis_rnn(
                    self.dis_emb_layer(utils.one_hot(sim_phrase,
                                                     self.vocab_sz)))[1])
            enc_out = self.dis_lin(
                self.dis_rnn(self.dis_emb_layer(torch.exp(out)))[1])

        enc_out.squeeze_(0)
        enc_sim_phrase.squeeze_(0)
        return out, enc_out, enc_sim_phrase


    def generateInfo(self):
        """
        Generates:
        vocab = {index : word} e.g. {1: "Hello"}    taken from data/Vocab_Extra
        vocab_embed = {word : embeddings} e.g. {"Hello" : [1, 2, 3 .... 511, 512]} currently taken from embeddings.p but will have to be fixeds
        tree = scipy spatial tree
        """
        with open('data/Vocab_Extra','rb') as f:
            self.vocab = pickle.load(f)

        with open('data/embeddings.p','rb') as f:
            self.vocab_embed = pickle.load(f)

        # Reverse vocab ---> {"Hello" : 1}
        self.reverse_vocab = {x[1]:x[0] for x in list(self.vocab.items())}

        # Generate a scipy try from the embeddings we have
        self.tree = spatial.KDTree(list(self.vocab_embed.values()))


    def find_closest_word(self, word):

        word = torch.Tensor([word])

        print(word.size)

        # Embed unknown word
        embedding = self.enc_lin(
                self.enc_rnn(
                    self.emb_layer(utils.one_hot(word, self.vocab_sz)))[1])
        
        print("Encoded unknown word {} to an embedding of size {}".format(word, embedding.shape))

        known_index = (self.tree.query(wordembedding)[1])

        print("Unknown word now converted to index {}, which corresponds to word {}".format(known_index, self.vocab[known_index]))



    def tokenize(self, sentence):
        return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];


    def expandToMax(self, sent):
        if len(sent) > self.max_seq_len:
            
            ### Deal with this by removing stop words ###
            
            print("Removing phrase, too long:   "+" ".join(sent))
            return None
        else:
            while len(sent) < self.max_seq_len:
                sent.append(" ")
            return sent

    
    def inputData(self, sents):

        # Tokenize the words, keeping all punctuation etc
        tokens = [self.tokenize(sent) for sent in sents]

        # sentencesT of shape (batch_size, max_seq_size)
        sentencesT = np.array([self.expandToMax(x) for x in tokens])
        # Sentences of shape (max_seq_size, batch_size)
        sentences = sentencesT.transpose()
        
        # Make a zero version of sentences
        out_set = np.zeros(sentences.shape)

    
        # Fill out_set with indices
        for i in range(sentences.shape[0]):
            for j in range(sentences.shape[1]):
                try:
                    out_set[i][j] = self.reverse_vocab[sentences[i][j]]
                except KeyError:
                    closest_word = self.find_closest_word(sentences[i][j])
                    out_set[i][j] = self.reverse_vocab[closest_word]
        
        return out_set

    def generate(self, sents):

        sents = self.inputData(sents)

        out, _, _ = self.forward(sents)

        return out


def generate(sents):
    
    op = {
    "vocab_sz": 27699,#data.getVocabSize(),
    "max_seq_len": 28,#data.getSeqLength(),
    "emb_hid_dim": 256,#args.emb_hid_dim,
    "emb_dim": 512,#args.emb_dim,
    "enc_dim": 512,#args.enc_dim,
    "enc_dropout": 0.5,#args.enc_dropout,
    "enc_rnn_dim": 512,#args.enc_rnn_dim,
    "gen_rnn_dim": 512,#args.gen_rnn_dim,
    "gen_dropout": 0.5 #args.gen_dropout,
    #"lr": 0.0008,#args.learning_rate,
    #"epochs": 1,#args.n_epoch
    }

    files = glob("save/*")
    files.sort(key=os.path.getmtime)
    WEIGHT_PATH = files[-1]

    model = ParaphraseGenerator(op)
    model.load_state_dict(torch.load(WEIGHT_PATH))

    print("### Loading weights from {} ###".format(WEIGHT_PATH))
    
    out = model.generate(sents)

    print(out)
    

if __name__ == "__main__":

    sents = ["This is a test sentence, used to test the model.", "The second sentence is a bit trickier, as it is less straight forwards.", "Hello friends, here is a real sounding made up word: hallo"]

    generate(sents)