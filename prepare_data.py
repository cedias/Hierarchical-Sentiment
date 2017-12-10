import spacy
import gzip
import argparse
import gensim
import logging
import json
import pickle as pkl

from tqdm import tqdm
from random import randint,shuffle
from collections import Counter, Iterator


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) #gensim logging

#Global dicts for users
USERS = {}
ITEMS = {}

def count_lines(file):
    count = 0
    for _ in file:
        count += 1
    file.seek(0)
    return count


def build_dataset(args):

    def preprocess(datas):
        for data in datas:

            uk = USERS.setdefault(data["reviewerID"], len(USERS))
            ik = ITEMS.setdefault(data["asin"], len(ITEMS))

            yield (uk,ik,data['reviewText'],max(1,int(round(float(data["overall"]))))-1) #zero is useless, classes between 0-4 for 1-5 reviews

    def preprocess_rescale(datas):
        for data in datas:

            uk = USERS.setdefault(data["reviewerID"], len(USERS))
            ik = ITEMS.setdefault(data["asin"], len(ITEMS))

            rating = max(1,int(round(float(data["overall"]))))-1

            if rating > 3:
                rating = 1
            elif rating == 3:
                yield None
                continue
            else:
                rating = 0
            yield (uk, ik, data['reviewText'],rating) #zero is useless

    def data_generator(data):
        with gzip.open(args.input,"r") as f:
            for x in tqdm(f,desc="Reviews",total=count_lines(f)):
                yield json.loads(x)

    class TokIt(Iterator):
        def __init__(self, tokenized):
            self.tok = tokenized
            self.x = 0
            self.stop = len(tokenized)

        def __iter__(self):
            return self

        def next(self):
            if self.x < self.stop:
                self.x += 1
                return list(w.orth_ for w in self.tok[self.x-1] if len(w.orth_.strip()) >= 1 ) #whitespace shouldn't be a word.
            else:
                self.x = 0
                raise StopIteration 
        __next__ = next




    print("Building dataset from : {}".format(args.input))
    print("-> Building {} random splits".format(args.nb_splits))

    nlp = spacy.load('en')

    tokenized = [tok for tok in tqdm(nlp.tokenizer.pipe((x["reviewText"] for x in data_generator(args.input)), batch_size=10000, n_threads=8),desc="Tokenizing")]

    

    if args.create_emb:
        w2vmodel = gensim.models.Word2Vec(TokIt(tokenized), size=args.emb_size, window=5, min_count=5, iter=args.epochs, max_vocab_size=args.dic_size, workers=4)
        print(len(w2vmodel.wv.vocab))
        w2vmodel.wv.save_word2vec_format(args.emb_file,total_vec=len(w2vmodel.wv.vocab))        

    if args.rescale:
        print("-> Rescaling data to 0-1 (3's are discarded)")
        data = [dt for dt in tqdm(preprocess_rescale(data_generator(args.input)),desc="Processing") if dt is not None]
    else:
        data = [dt for dt in tqdm(preprocess(data_generator(args.input)),desc="Processing")]

    shuffle(data)

    splits = [randint(0,args.nb_splits-1) for _ in range(0,len(data))]
    count = Counter(splits)

    print("Split distribution is the following:")
    print(count)

    return {"data":data,"splits":splits,"rows":("user_id","item_id","review","rating")}


def main(args):
    ds = build_dataset(args)
    pkl.dump(ds,open(args.output,"wb"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str, default="sentences.pkl")
    parser.add_argument("--rescale",action="store_true")
    parser.add_argument("--nb_splits",type=int, default=5)

    parser.add_argument("--create-emb",action="store_true")
    parser.add_argument("--emb-file", type=str, default=None)
    parser.add_argument("--emb-size",type=int, default=100)
    parser.add_argument("--dic-size", type=int,default=10000000)
    parser.add_argument("--epochs", type=int,default=1)
    args = parser.parse_args()

    if args.emb_file is None:
        args.emb_file = args.output + "_emb.txt"

    main(args)
