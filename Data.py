import spacy

from collections import Counter
from operator import itemgetter
from collections import OrderedDict
from random import choice, shuffle
from tqdm import tqdm

import torch
import torch.utils.data as data
import torch.nn.functional as fn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler


class TuplesListDataset(Dataset):

    def __init__(self, tuplelist,rows):
        super(TuplesListDataset, self).__init__()
        self.tuplelist = tuplelist
        self.mappings = {}
        self.rows = rows

        print(rows)
    def __len__(self):
        return len(self.tuplelist)

    def __getitem__(self,index):
        if len(self.mappings) == 0:
            return self.tuplelist[index]
        else:
            t = list(self.tuplelist[index])

            for i,m in self.mappings.items():
                t[i] = m(t[i])

            return tuple(t)

    def __iter__(self):
        return self.tuplelist.__iter__()

    def _f2i(self,field):
        if type(field) == int:
            return field
        if type(field) == str:
            return self.rows[field]

        raise IndexError("field {} doesn't exist".format(field))

    def field_gen(self,field,transform=False):
        field = self._f2i(field)
        if transform:
            for i in range(len(self)):
                yield self[i][field]
        else:
            for x in self:
                yield x[field]


    def get_stats(self,field):
        field = self._f2i(field)
        d =  dict(Counter(self.field_gen(field)))
        sumv = sum([v for k,v in d.items()])
        class_per = {k:(v/sumv) for k,v  in d.items()}

        return d,class_per

    def get_field_dict(self,field,offset=0):
        field = self._f2i(field)
        d2k = {c:i for i,c in enumerate(set(self.field_gen(field)),offset)}
        return d2k

    def set_mapping(self,field,mapping=None,offset=0, unk=None):
        """
        Sets or creates a mapping for a tuple field. Mappings are {k:v} with keys starting at offset.
        """
        field = self._f2i(field)
        if mapping is None:
            mapping = self.get_field_dict(field,offset)

        else:
            if unk is not None:
                mapping.update(((uk,unk) for uk in set(self.field_gen(field)) if uk not in mapping))
            
        self.mappings[field] = mapping.__getitem__

        return mapping

    def set_transform(self,field,transform):
        """
        sets a field transformation where transform is a function of the field i.e f(field) -> transformed
        """
        field = self._f2i(field)
        self.mappings[field] = transform

    def prebuild(self):
        return [self[i] for i in tqdm(range(len(self)),total=len(self),desc="Prebuilding set")]

    @staticmethod
    def build_train_test(datatuples,splits,split_num=0,validation=0.5,rows=None):
        """
        Builds train/val/test sets
        Validation set at 0.5 if n split is 5 gives an 80:10:10 split as usually used.
        """
        train,test = [],[]

        for split,data in tqdm(zip(splits,datatuples),total=len(datatuples),desc="Building train/test of split #{}".format(split_num)):
            if split == split_num:
                test.append(data)
            else:
                train.append(data)


        if len(test) <= 0:
                raise IndexError("Test set is empty - split {} probably doesn't exist".format(split_num))

        if rows and type(rows) is tuple:
            rows = {v:k for k,v in enumerate(rows)}
            print("TuplesListDataset rows are the following:")
            print(rows)

        if validation > 0:

            if 0 < validation < 1:
                val_len = int(validation * len(test))

            validation = test[-val_len:]
            test = test[:-val_len]


            return TuplesListDataset(train,rows),TuplesListDataset(validation,rows),TuplesListDataset(test,rows)

        return TuplesListDataset(train,rows),None,TuplesListDataset(test,rows) #None for no pb
        


class BucketSampler(Sampler):
    """
    Evenly sample from bucket for datalen
    """

    def __init__(self, dataset,field):
        self.dataset = dataset
        self.field = field
        self.index_buckets = self._build_index_buckets()
        self.len = min([len(x) for x in self.index_buckets.values()])

    def __iter__(self):
        return iter(self.bucket_iterator())

    def __len__(self):
        return self.len

    def bucket_iterator(self):
        cl = list(self.index_buckets.keys())
   
        for x in range(len(self)):
            yield choice(self.index_buckets[choice(cl)])

            
    def _build_index_buckets(self):
        class_index = {}
        for ind,cl in enumerate(self.dataset.field_gen(self.field,True)):
            if cl not in class_index:
                class_index[cl] = [ind]
            else:
                class_index[cl].append(ind)
        return class_index
        



class Vectorizer():

    def __init__(self,word_dict=None,max_sent_len=8,max_word_len=32):
        self.word_dict = word_dict
        self.nlp = spacy.load('en')
        self.max_sent_len = max_sent_len
        self.max_word_len = max_word_len


    def _get_words_dict(self,data,max_words):
        word_counter = Counter(w.lower_ for d in self.nlp.tokenizer.pipe((doc for doc in tqdm(data,desc="Tokenizing data"))) for w in d)
        dict_w =  {w: i for i,(w,_) in tqdm(enumerate(word_counter.most_common(max_words),start=2),desc="building word dict",total=max_words)}
        dict_w["_padding_"] = 0
        dict_w["_unk_word_"] = 1
        print("Dictionnary has {} words".format(len(dict_w)))
        return dict_w

    def build_dict(self,text_iterator,max_f):
        self.word_dict = self._get_words_dict(text_iterator,max_f)

    def vectorize_batch(self,t,trim=True):
        return self._vect_dict(t,trim)


    def _vect_dict(self,t,trim):

        if self.word_dict is None:
            print("No dictionnary to vectorize text \n-> call method build_dict \n-> or set a word_dict attribute \n first")
            raise Exception

        if type(t) == str:
            t = [t]

        revs = []
        
        for rev in t:
            review = []
            for j,sent in enumerate(self.nlp(rev).sents):  

                if trim and j>= self.max_sent_len:
                    break
                s = []
                for k,word in enumerate(sent):
                    word = word.lower_

                    if trim and k >= self.max_word_len:
                        break

                    if word in self.word_dict:
                        s.append(self.word_dict[word])
                    else:
                        s.append(self.word_dict["_unk_word_"]) #_unk_word_
                 
                if len(s) >= 1:
                    review.append(s)
            if len(review) == 0:
                review = [[self.word_dict["_unk_word_"]]]        
            revs.append(review)

        return revs

