#utils.py
import torch
from tqdm import tqdm
from torch.autograd import Variable
from fmtl import FMTL


def tuple2var(tensors,data):
    def copy2tensor(t,data):
        t.resize_(data.size()).copy_(data,async=True)
        return Variable(t)
    return tuple(map(copy2tensor,tensors,data))

def new_tensors(n,cuda,types={}):
    def new_tensor(t_type,cuda):
        x = torch.Tensor()
        if t_type:
            x = x.type(t_type)
        if cuda:
            x = x.cuda()
        return x
    return tuple([new_tensor(types.setdefault(i,None),cuda) for i in range(0,n)])

def accuracy(out,truth):
    def sm(mat):
        exp = torch.exp(mat)
        sum_exp = exp.sum(1,True)+0.0001
        return exp/sum_exp.expand_as(exp)

    _,max_i = torch.max(sm(out),1)

    eq = torch.eq(max_i,truth).float()
    all_eq = torch.sum(eq)

    return all_eq, all_eq/truth.size(0)*100, max_i.float()

def checkpoint(epoch,net,output):
    model_out_path = output+"_epoch_{}.pth".format(epoch)
    torch.save(net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def load_embeddings(file,offset=0):
    emb_file = open(file).readlines()
    first = emb_file[0]
    word, vec = int(first.split()[0]),int(first.split()[1])
    size = (word,vec)
    print("--> Got {} words of {} dimensions".format(size[0],size[1]))
    tensor = torch.zeros(size[0]+offset,size[1]) ## adding offset
    word_d = {}

    print("--> Shape with padding and unk_token:")
    print(tensor.size())

    for i,line in tqdm(enumerate(emb_file),desc="Creating embedding tensor",total=len(emb_file)):
        if i==0: #skipping header (-1 to the enumeration to take it into account)
            print("skipping embedding size line:\'{}\'".format(line.strip()))
            continue

        spl = line.strip().split(" ")

        if len(spl[1:]) == size[1]: #word is most probably whitespace or junk if badly parsed
            word_d[spl[0]] = i + offset-1
            tensor[i+offset-1] = torch.FloatTensor(list(float(x) for x in spl[1:]))
        else:
            print("WARNING: MALFORMED EMBEDDING DICTIONNARY:\n {} \n line isn't parsed correctly".format(line))

    try:
        assert(len(word_d)==size[0])
    except:
        print("Final dictionnary length differs from number of embeddings - some lines were malformed.")

    return tensor, word_d



def FMTL_train_val_test(datatuples,splits,split_num=0,validation=0.5,rows=None):
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

    idxs = (train,test,validation)
    fmtl = FMTL(datatuples,rows)
    iters = idxs

    return (fmtl,iters)