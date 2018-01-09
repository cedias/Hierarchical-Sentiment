from collections import Counter
from operator import itemgetter
from collections import OrderedDict
from random import choice, shuffle
from tqdm import tqdm
import itertools

import torch
import torch.utils.data as data
import torch.nn.functional as fn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler



class FMTL_iterator(Dataset):
    """
    Simple indexed iterator on FMTL
    """
    def __init__(self,fmtl,idxs):
        super(FMTL_iterator, self).__init__()
        self.fmtl = fmtl
        self.idxs = idxs
        
    def __getitem__(self,i):
        return self.fmtl[i]
    
    def __len__(self):
        return len(self.idxs)

    def __iter__(self):
        self.iter_idx = self.idxs.__iter__()
        return self
    
    def __next__(self):
        idx = self.iter_idx.__next__()
        return self.fmtl[idx]

    def prebuild(self):
        return [x for x in tqdm(self,desc="Prebuilding")]
        
    @staticmethod
    def from_seq(fmtl,idxs):
        return tuple([FMTL_iterator(fmtl,idx) for idx in idxs])
        

class FMTL():
    
    """
    A field-mappable tuple list
    Each field's atomic unit can be mapped to a value. 
    """

    def __init__(self, tuplelist,rows=None):
        self.tuplelist = tuplelist
        self.mappings = {}
        self.unknown = {}
        self.rows = rows
       
    def __len__(self):
        return len(self.tuplelist)

    def __getitem__(self,index):
        """
        Maps field[x]:
            -> if field is a tuple/list maps each element inside, keeping structure
            -> else directly maps
            -> if mapping function return error, tries to map with 'unk' value.
            (see self._rec_apply)
        """
        if len(self.mappings) == 0:
            return self.tuplelist[index]
        else:
            t = list(self.tuplelist[index])
            for i,m in self.mappings.items():
                t[i] = self._rec_apply(m,t[i],self.unknown.get(i))

            return tuple(t)

    def __iter__(self):
        self.iter_idxs = range(len(self.tuplelist)).__iter__()
        return self

    def __next__(self):
        return self[self.iter_idxs.__next__()]

    def _f2i(self,field):
        if type(field) == int:
            return field

        if type(field) == str:
            return self.rows[field]

        if type(field) == str and self.rows == None:
            raise IndexError("field {} index is unknown, no rows attribute provided".format(field))

        raise IndexError("field {} doesn't exist".format(field))
    
    def _dic_or_fun(self,mapper):
        if type(mapper) is dict:
            return mapper.__getitem__
        else:
            return mapper

    def _rec_apply(self,f,item,unk=None):
        if isinstance(item,list) or isinstance(item,tuple):
            return type(item)(map(lambda x:self._rec_apply(f,x,unk), item))
        else:
            try:
                return f(item)
            except:
                if unk is not None:
                    return unk
                else:
                    raise KeyError("No mapping or placeholder for value: {}".format(item))

        
    def set_mapping(self, field, mapping, unk=None):
        """
        Sets a mapping for a tuple field. Mappings are functions of a field value or dict
        """
        field = self._f2i(field)
        self.mappings[field] = self._dic_or_fun(mapping)

        if unk is not None:
            self.unknown[field] = unk

        return mapping

    def field_gen(self, field, key_iter=None):
        field = self._f2i(field)
        if key_iter is not None:
            for x in key_iter:
                yield x[field]
        else:
            for x in self:
                yield x[field]

    def get_stats(self, field, key_iter=None, verbose=False):
        """
        helper for stats on a field. Field values should be hashable.
        """
        field = self._f2i(field)
        d =  dict(Counter(self.field_gen(field,key_iter=key_iter)))
        sumv = sum([v for k,v in d.items()])
        class_per = {k:(v/sumv) for k,v  in d.items()}

        if verbose:
            print(d)
            print(class_per)

        return d,class_per

    def get_field_dict(self, field, offset=0, max_count=-1, key_iter=None, iter_func=None):
        """
        Helper to create a dict from a field.
        """
        field_gen = self.field_gen(self._f2i(field),key_iter=key_iter)

        if iter_func is not None:
            field_gen = itertools.chain.from_iterable((x for x in iter_func(field_gen)))
            
        d =  Counter(field_gen)

        if max_count > -1:
            d_keys = (k for k,v in d.most_common(max_count))
        else:
            d_keys = d.keys()
    
        d2k = {c:i for i,c in enumerate(d_keys,offset)}
        return d2k

    def prebuild(self):
        """
        Freeze mappings into a new FMTL instance.
        """
        return FMTL([x for x in tqdm(self,desc="Prebuilding")],self.rows)

    @staticmethod
    def from_list(datatuples,splits,split_num=0,validation=0.5,rows=None):
        """
        Builds train/val/test indexes sets from tuple list and split list
        Validation set at 0.5 if n split is 5 gives an 80:10:10 split as usually used.
        """
        train,test = [],[]

        for idx,split in tqdm(enumerate(splits),total=len(splits),desc="Building train/test of split #{}".format(split_num)):
            if split == split_num:
                test.append(idx)
            else:
                train.append(idx)

        if len(test) <= 0:
                raise IndexError("Test set is empty - split {} probably doesn't exist".format(split_num))

        if rows and type(rows) is tuple:
            rows = {v:k for k,v in enumerate(rows)}
            print("Tuples rows are the following:")
            print(rows)

        if validation > 0:

            if 0 < validation < 1:
                val_len = int(validation * len(test))

            validation = test[-val_len:]
            test = test[:-val_len]

        else:
            validation = []

        idxs = [train,test,validation]
        fmtl = FMTL(datatuples,rows)
        iters = FMTL_iterator.from_seq(fmtl,idxs)

        return (fmtl,iters)

