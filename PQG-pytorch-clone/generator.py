import numpy as np
from models.enc_dec_dis import ParaphraseGenerator
import torch
import torch.nn as nn
from glob import glob
import os
from misc import net_utils, utils
import re
import pickle
import spacy
from scipy import spatial

def find_closest_word(word, vocab_embed, tree):
    word = nlp(str(word))
    wordembedding = word[0].vector
    
    
    
    index_embeddings = tree.query(wordembedding)[1]
    
    # embedded_words = {"word": embedding vector}
    index_embedded_words = list(vocab_embed.keys())[index_embeddings]
    
    return vocab[index_embedded_words]


def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def expandToMax(sent, max_sent_size):
    if len(sent) > max_sent_size:
        print("Removing phrase, too long:   "+" ".join(sent))
        return None
    else:
        while len(sent) < max_sent_size:
            sent.append(" ")
        return sent

    
def inputData(sents, max_sent_size, vocab, vocab_embed, tree):

    # Remove 
    tokens = [tokenize(sent) for sent in sents]

    # sentencesT of shape (batch_size, max_seq_size)
    sentencesT = np.array([expandToMax(x, max_sent_size) for x in tokens])
    # Sentences of shape (max, batch)
    sentences = sentencesT.transpose()
    
    out_set = np.zeros(sentences.shape)

    reverse_vocab = {x[1]:x[0] for x in list(vocab.items())}
   
    
    for i in range(sentences.shape[0]):
        for j in range(sentences.shape[1]):
            try:
                out_set[i][j] = reverse_vocab[sentences[i][j]]
            except KeyError:
                closest_word = find_closest_word(sentences[i][j],vocab, tree)
                out_set[i][j] = reverse_vocab[closest_word]
    
    return out_set

with open('data/Vocab_Extra','rb') as f:
    vocab = pickle.load(f)

embeds = {}
import en_core_web_sm
nlp = en_core_web_sm.load()

#for word in list(vocab.values()):
#    doc = nlp(word)
#    embedding = doc[0].vector
#    embeds[word] = embedding
    
#with open('embeddings.p', 'wb') as fp:
#    pickle.dump(embeds, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('embeddings.p','rb') as f:
    embeds = pickle.load(f)
    
tree = spatial.KDTree(np.array(list(embeds.values())))

def main(sents):

    parser = utils.make_parser()
    args = parser.parse_args()

    # build model

    # # get data
    #data = Dataloader(args.input_json, args.input_ques_h5)

    # # make op
    op = {
        "vocab_sz": 27699,#data.getVocabSize(),
        "max_seq_len": 28,#data.getSeqLength(),
        "emb_hid_dim": 256,#args.emb_hid_dim,
        "emb_dim": 512,#args.emb_dim,
        "enc_dim": 512,#args.enc_dim,
        "enc_dropout": 0.5,#args.enc_dropout,
        "enc_rnn_dim": 512,#args.enc_rnn_dim,
        "gen_rnn_dim": 512,#args.gen_rnn_dim,
        #"gen_dropout": 0.5,#args.gen_dropout,
        #"lr": 0.0008,#args.learning_rate,
        #"epochs": 1,#args.n_epoch
    }

    files = glob("save/*")
    files.sort(key=os.path.getmtime)
    WEIGHT_PATH = files[-1]
    print("### Loading weights from {} ###".format(WEIGHT_PATH))

    model = ParaphraseGenerator(op)
    model.load_state_dict(torch.load(WEIGHT_PATH))

    print("Maximum sequence length = {}".format(28))
    
    with open('data/Vocab_Extra','rb') as f:
        vocab = pickle.load(f)
        
    
    sents = inputData(sents, 28, vocab, embed_model)

    out, _, _ = model.forward(sents)