import pickle as pkl
import gensim
import argparse
import logging
from utils import *
import itertools

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) #gensim logging


class Word_Iterator:
    def __init__(self, fmtl,train_idx):
        self.fmtl = fmtl
        self.train_idx = train_idx
        self.x = 0
        self.stop = len(train_idx)

    def __iter__(self):
        self.field_gen = self.fmtl.field_gen("review",self.train_idx)
        return self

    def next(self):
        try:
            return list(x for x in itertools.chain.from_iterable(self.field_gen.__next__()) if len(x.strip())>0) #whitespace isn't something we want.
        except:
            raise StopIteration 
    __next__ = next


def build_save(datadict,args):
    data_tl,(trainit,valit,testit) = FMTL_train_val_test(datadict["data"],datadict["splits"],args.split,validation=0.5,rows=datadict["rows"])
    word_it = Word_Iterator(data_tl,trainit)

    w2vmodel = gensim.models.Word2Vec(word_it, size=args.emb_size, window=args.window, min_count=args.min_count, iter=args.epochs, max_vocab_size=args.dic_size, workers=args.threads)
    print(len(w2vmodel.wv.vocab))

    if args.output is None:
        out_file = args.filename+"_w2v_s{}.txt".format(args.split)

    w2vmodel.wv.save_word2vec_format(out_file,total_vec=len(w2vmodel.wv.vocab))  



def main_func(args):
    datadict = pkl.load(open(args.filename,"rb"))

    if args.split < 0:
        print("==> Building embeddings for each splits")
        for split in set(datadict["splits"]):
            build_save(datadict,args)
    else:
        print("==> Building embeddings for split {}".format(args.split))
        build_save(datadict,args)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument("--output", type=str,default=None)
    parser.add_argument("--emb_size",type=int,default=200)
    parser.add_argument("--dic-size", type=int,default=10000000)
    parser.add_argument("--epochs", type=int,default=1)
    parser.add_argument("--min-count", type=int,default=5)
    parser.add_argument("--threads", type=int,default=4)
    parser.add_argument("--window", type=int,default=5)
    parser.add_argument("--split", type=int, default=0)

    args = parser.parse_args()


    main_func(args)

