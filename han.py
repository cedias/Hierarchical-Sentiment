import argparse
import pickle as pkl
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
from Nets import NSCUPA, HAN
from Data import TuplesListDataset, Vectorizer
from fmtl import FMTL
from utils import *
import sys
import quiviz
import logging



def save(net,dic,path):
    """
    Saves a model's state and it's embedding dic by piggybacking torch's save function
    """
    dict_m = net.state_dict()
    dict_m["word_dic"] = dic    
    torch.save(dict_m,path)


def tuple_batch(l):
    """
    Prepare batch
    - Reorder reviews by length
    - Split reviews by sentences which are reordered by length
    - Build sentence ordering index to extract each sentences in training loop
    """
    _,_,review,rating = zip(*l)
    r_t = torch.Tensor(rating).long()
    list_rev = review

    sorted_r = sorted([(len(r),r_n,r) for r_n,r in enumerate(list_rev)],reverse=True) #index by desc rev_le
    lr,r_n,ordered_list_rev = zip(*sorted_r)

    max_sents = lr[0]

    #reordered
    r_t = r_t[[r_n]]
    review = [review[x] for x in r_n]

    stat =  sorted([(len(s),r_n,s_n,s) for r_n,r in enumerate(ordered_list_rev) for s_n,s in enumerate(r)],reverse=True)
    max_words = stat[0][0]

    ls = []
    batch_t = torch.zeros(len(stat),max_words).long()                         # (sents ordered by len)
    sent_order = torch.zeros(len(ordered_list_rev),max_sents).long().fill_(0) # (rev_n,sent_n)

    for i,s in enumerate(stat):
        sent_order[s[1],s[2]] = i+1 #i+1 because 0 is for empty.
        batch_t[i,0:len(s[3])] = torch.LongTensor(s[3])
        ls.append(s[0])

    return batch_t,r_t,sent_order,ls,lr,review

@quiviz.log
def train(epoch,net,optimizer,dataset,criterion,cuda):
    net.train()
    epoch_loss = 0
    mean_mse = 0
    mean_rmse = 0
    ok_all = 0
    data_tensors = new_tensors(3,cuda,types={0:torch.LongTensor,1:torch.LongTensor,2:torch.LongTensor}) #data-tensors

    with tqdm(total=len(dataset),desc="Training") as pbar:
        for iteration, (batch_t,r_t,sent_order,ls,lr,review) in enumerate(dataset):

            data = tuple2var(data_tensors,(batch_t,r_t,sent_order))
            optimizer.zero_grad()
            out = net(data[0],data[2],ls,lr)

            ok,per,val_i = accuracy(out,data[1])
            ok_all += per.data[0]

            mseloss = F.mse_loss(val_i,data[1].float())
            mean_rmse += math.sqrt(mseloss.data[0])
            mean_mse += mseloss.data[0]
            loss =  criterion(out, data[1]) 
            epoch_loss += loss.data[0]
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix({"acc":ok_all/(iteration+1),"CE":epoch_loss/(iteration+1),"mseloss":mean_mse/(iteration+1),"rmseloss":mean_rmse/(iteration+1)})

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}, {}% accuracy".format(epoch, epoch_loss /len(dataset),ok_all/len(dataset)))
    return {"train_acc": ok_all/len(dataset)}

@quiviz.log
def test(epoch,net,dataset,cuda,msg="test"):
    net.eval()
    epoch_loss = 0
    ok_all = 0
    pred = 0
    skipped = 0
    mean_mse = 0
    mean_rmse = 0
    data_tensors = new_tensors(3,cuda,types={0:torch.LongTensor,1:torch.LongTensor,2:torch.LongTensor}) #data-tensors
    
    with tqdm(total=len(dataset),desc=msg) as pbar:
        for iteration, (batch_t,r_t,sent_order,ls,lr,review) in enumerate(dataset):
            data = tuple2var(data_tensors,(batch_t,r_t,sent_order))
            out  = net(data[0],data[2],ls,lr)
            ok,per,val_i = accuracy(out,data[1])


            mseloss = F.mse_loss(val_i,data[1].float())
            mean_rmse += math.sqrt(mseloss.data[0])
            mean_mse += mseloss.data[0]

            ok_all += per.data[0]
            pred+=1

            pbar.update(1)
            pbar.set_postfix({"acc":ok_all/pred, "skipped":skipped,"mseloss":mean_mse/(iteration+1),"rmseloss":mean_rmse/(iteration+1)})


    print("===> {} Complete:  {}% accuracy".format(msg,ok_all/pred))
    return {f"{msg}_acc": ok_all/pred}


def load(args):

    datadict = pkl.load(open(args.filename,"rb"))
    data_tl,(trainit,valit,testit) = FMTL_train_val_test(datadict["data"],datadict["splits"],args.split,validation=0.5,rows=datadict["rows"])

    rating_mapping = data_tl.get_field_dict("rating",key_iter=trainit) #creates class mapping
    data_tl.set_mapping("rating",rating_mapping) 

    if args.load:
        state = torch.load(args.load)
        wdict = state["word_dic"]
    else:
        if args.emb:
            tensor,wdict = load_embeddings(args.emb,offset=2)
        else:     
            wdict = data_tl.get_field_dict("review",key_iter=trainit,offset=2, max_count=args.max_feat, iter_func=(lambda x: (w for s in x for w in s )))

        wdict["_pad_"] = 0
        wdict["_unk_"] = 1
    
    if args.max_words > 0 and args.max_sents > 0:
        print("==> Limiting review and sentence length: ({} sents of {} words) ".format(args.max_sents,args.max_words))
        data_tl.set_mapping("review",(lambda f:[[wdict.get(w[:args.max_words],1) for w in s[:args.max_sents]] for s in f]))
    else:
        data_tl.set_mapping("review",wdict,unk=1)


    print("Train set class stats:\n" + 10*"-")
    _,_ = data_tl.get_stats("rating",trainit,True)

    if args.load:
        #print(state.keys())
        net = HAN(ntoken=len(state["word_dic"]),emb_size=state["embed.weight"].size(1),hid_size=state["sent.rnn.weight_hh_l0"].size(1),num_class=state["lin_out.weight"].size(0))
        del state["word_dic"]
        net.load_state_dict(state)

    else:
        if args.emb:
            net = HAN(ntoken=len(wdict),emb_size=len(tensor[1]),hid_size=args.hid_size,num_class=len(rating_mapping))
            net.set_emb_tensor(torch.FloatTensor(tensor))
        else:
            net = HAN(ntoken=len(wdict), emb_size=args.emb_size,hid_size=args.hid_size, num_class=len(rating_mapping))

    if args.prebuild:
        data_tl = FMTL(list(x for x  in tqdm(data_tl,desc="prebuilding")),data_tl.rows)

    return data_tl,(trainit,valit,testit), net, wdict


def main(args):

    print(32*"-"+"\nHierarchical Attention Network:\n" + 32*"-")
    data_tl, (train_set, val_set, test_set), net, wdict = load(args)


    dataloader = DataLoader(data_tl.indexed_iter(train_set), batch_size=args.b_size, shuffle=True, num_workers=3, collate_fn=tuple_batch,pin_memory=True)
    dataloader_valid = DataLoader(data_tl.indexed_iter(val_set), batch_size=args.b_size, shuffle=False,  num_workers=3, collate_fn=tuple_batch)
    dataloader_test = DataLoader(data_tl.indexed_iter(test_set), batch_size=args.b_size, shuffle=False, num_workers=3, collate_fn=tuple_batch,drop_last=True)

    criterion = torch.nn.CrossEntropyLoss()      

    if args.cuda:
        net.cuda()

    print("-"*20)

    last_acc = 0

    optimizer = optim.Adam(net.parameters())
    torch.nn.utils.clip_grad_norm(net.parameters(), args.clip_grad)

    for epoch in range(1, args.epochs + 1):
        print("\n-------EPOCH {}-------".format(epoch))
        logging.info("\n-------EPOCH {}-------".format(epoch))

        train(epoch,net,optimizer,dataloader,criterion,args.cuda)

        if args.snapshot:
            print("snapshot of model saved as {}".format(args.save+"_snapshot"))
            save(net,wdict,args.save+"_snapshot")

        val = test(epoch,net,dataloader_valid,args.cuda,msg="val")
        new_acc = list(val.values())[0]

        if new_acc < last_acc:
            logging.info("---- EARLY STOPPING")
            sys.exit()
        last_acc = new_acc

        test(epoch,net,dataloader_test,args.cuda)

    if args.save:
        print("model saved to {}".format(args.save))
        save(net,wdict,args.save)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hierarchical Attention Networks for Document Classification')
    
    parser.add_argument("--emb-size",type=int,default=200)
    parser.add_argument("--hid-size",type=int,default=100)

    parser.add_argument("--max-feat", type=int,default=10000)
    parser.add_argument("--epochs", type=int,default=10)
    parser.add_argument("--clip-grad", type=float,default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum",type=float,default=0.9)
    parser.add_argument("--b-size", type=int, default=32)

    parser.add_argument("--emb", type=str)
    parser.add_argument("--max-words", type=int,default=-1)
    parser.add_argument("--max-sents",type=int,default=-1)

    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--load", type=str)
    parser.add_argument("--save", type=str)
    parser.add_argument("--snapshot", action='store_true')
    parser.add_argument("--prebuild",action="store_true")
    parser.add_argument('--cuda', action='store_true', help='use CUDA')

    parser.add_argument("--output", type=str)
    parser.add_argument('filename', type=str)
    args = parser.parse_args()
    logging.info("========== NEW XP =============")
    main(args)
