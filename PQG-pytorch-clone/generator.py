import numpy as np
from models.enc_dec_dis import ParaphraseGenerator
import torch
import torch.nn as nn
from glob import glob
import os
import nltk
from misc import net_utils, utils
#from misc.dataloader import Dataloader


def expandToMax(sent, max_sent_size):
    if len(sent) > max_sent_size:
        print("Removing phrase, too long:   "+" ".join(sent))
    else:
        while len(sent) < max_sent_size:
            sent.append(" ")
        
        return sent

def inputData(sents, max_sent_size):

    # Remove 
    tokens = [nltk.word_tokenize(sent) for sent in sents]

    sentencesT = np.array([expandToMax(x, max_sent_size) for x in tokens])
    print(sentencesT)
    sentences = sentencesT.transpose()
    print(sentences)
    
    return sentences

def main():

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
        "gen_dropout": 0.5,#args.gen_dropout,
        "lr": 0.0008,#args.learning_rate,
        "epochs": 1,#args.n_epoch
    }
    print(op)

    files = glob("save/*")
    files.sort(key=os.path.getmtime)
    WEIGHT_PATH = files[-1]
    print("### Loading weights from {} ###".format(WEIGHT_PATH))

    model = ParaphraseGenerator(op)
    model.load_state_dict(torch.load(WEIGHT_PATH))

    print("Maximum sequence length = {}".format(28))

    sents = ["This is a test sentence, used to test the model.", "The second sentence is a bit trickier, as it is less straight forwards."]
    sents = inputData(sents, 28)

    out, _, _ = model.forward(sents)

if __name__ == "__main__":

    main()
