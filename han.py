import argparse
import pickle as pkl
import numpy as np
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
from DataNew import FMTL
from utils import *
import sys


def save(net,dic,path):
    dict_m = net.state_dict()
    dict_m["word_dic"] = dic    
    torch.save(dict_m,path)


def tuple_batch(l):
    _,_,review,rating = zip(*l)
    
    r_t = torch.Tensor(rating).long()
    list_rev = review

    sorted_r = sorted([(len(r),r_n,r) for r_n,r in enumerate(list_rev)],reverse=True) #index by desc rev_le
    lr,r_n,ordered_list_rev = zip(*sorted_r)

    max_sents = lr[0]
    #reordered
    r_t = r_t[[r_n]]
    review = [review[x] for x in r_n] #reorder reviews
    stat =  sorted([(len(s),r_n,s_n,s) for r_n,r in enumerate(ordered_list_rev) for s_n,s in enumerate(r)],reverse=True)
    max_words = stat[0][0]

    ls = []
    batch_t = torch.zeros(len(stat),max_words).long()                          # (sents ordered by len)
    sent_order = torch.zeros(len(ordered_list_rev),max_sents).long().fill_(0) # (rev_n,sent_n)

    for i,s in enumerate(stat):
        sent_order[s[1],s[2]] = i+1 #i+1 because 0 is for empty.
        batch_t[i,0:len(s[3])] = torch.LongTensor(s[3])
        ls.append(s[0])

    return batch_t,r_t,sent_order,ls,lr,review


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



def test(epoch,net,dataset,cuda,msg="Evaluating"):
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




def load(args):

    if not args.load:
        print("\nLoading Data:\n" + 25*"-")
        max_features = args.max_feat
        datadict = pkl.load(open(args.filename,"rb"))
        tuples = datadict["data"]
        splits  = datadict["splits"]
        rows = datadict["rows"]
        split_keys = set(x for x in splits)

        if args.split not in split_keys:
            print("Chosen split (#{}) not in split set {}".format(args.split,split_keys))
        else:
            print("Split #{} chosen".format(args.split))


        data_tl,(trainit,valit,testit) = FMTL.from_list(tuples,splits,args.split,validation=0.5,rows=rows)

        print("Train set length:",len(trainit))
        print("Test set length:",len(testit))

        rating_mapping = data_tl.get_field_dict("rating",key_iter=trainit) #creates class mapping
        data_tl.set_mapping("rating",rating_mapping) 

        if args.emb:
            tensor,dic = load_embeddings(args.emb,offset=2)
        else:     
            wdict = data_tl.get_field_dict("review",key_iter=trainit,offset=2, max_count=args.max_feat, iter_func=(lambda x: (w for s in x for w in s )))

        wdict["_pad_"] = 0
        wdict["_unk_"] = 1
        
        data_tl.set_mapping("review",wdict,unk=1)

        print("Train set:\n" + 10*"-")
        class_stats,class_per = data_tl.get_stats("rating",trainit,True)

        print(10*"-" + "\n Validation set:\n" + 10*"-")
        val_stats,val_per = data_tl.get_stats("rating",valit,True)


        if args.emb:
            net = HAN(ntoken=len(dic),emb_size=len(tensor[1]),hid_size=args.hid_size,num_class=len(rating_mapping))
            net.set_emb_tensor(torch.FloatTensor(tensor))
        else:
            net = HAN(ntoken=len(wdict), emb_size=args.emb_size,hid_size=args.hid_size, num_class=len(rating_mapping))

        trainit = trainit.prebuild()
        testit = testit.prebuild()
        valit = valit.prebuild()

    else:
        state = torch.load(args.load)
        net = HAN(ntoken=len(state["word_dic"]),emb_size=state["embed.weight"].size(1),hid_size=state["sent.gru.weight_hh_l0"].size(1),num_class=state["lin_out.weight"].size(0))
        del state["word_dic"]
        net.load_state_dict(state)

    return trainit,valit,testit,net



def main(args):

    print(32*"-"+"\nHierarchical Attention Network:\n" + 32*"-")
    train_set, val_set, test_set, net = load(args)

    dataloader = DataLoader(train_set, batch_size=args.b_size, shuffle=True, num_workers=3, collate_fn=tuple_batch,pin_memory=True)
    dataloader_valid = DataLoader(val_set, batch_size=args.b_size, shuffle=False,  num_workers=3, collate_fn=tuple_batch)
    dataloader_test = DataLoader(test_set, batch_size=args.b_size, shuffle=False, num_workers=3, collate_fn=tuple_batch,drop_last=True)

    criterion = torch.nn.CrossEntropyLoss()      

    if args.cuda:
        net.cuda()

    print("-"*20)

    optimizer = optim.Adam(net.parameters())
    torch.nn.utils.clip_grad_norm(net.parameters(), args.clip_grad)

    for epoch in range(1, args.epochs + 1):
        print("\n-------EPOCH {}-------".format(epoch))
        train(epoch,net,optimizer,dataloader,criterion,args.cuda)

        if args.snapshot:
            print("snapshot of model saved as {}".format(args.save+"_snapshot"))
            save(net,vectorizer.word_dict,args.save+"_snapshot")

        test(epoch,net,dataloader_valid,args.cuda,msg="Validation")
        test(epoch,net,dataloader_test,args.cuda)


    if args.save:
        print("model saved to {}".format(args.save))
        save(net,vectorizer.word_dict,args.save)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hierarchical Attention Networks for Document Classification')
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--emb-size",type=int,default=200)
    parser.add_argument("--hid-size",type=int,default=100)
    parser.add_argument("--b-size", type=int, default=32)
    parser.add_argument("--max-feat", type=int,default=10000)
    parser.add_argument("--epochs", type=int,default=10)
    parser.add_argument("--clip-grad", type=float,default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max-words", type=int,default=32)
    parser.add_argument("--max-sents",type=int,default=32)
    parser.add_argument("--momentum",type=float,default=0.9)
    parser.add_argument("--emb", type=str)
    parser.add_argument("--load", type=str)
    parser.add_argument("--save", type=str)
    parser.add_argument("--snapshot", action='store_true')
    parser.add_argument("--output", type=str)
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('filename', type=str)
    args = parser.parse_args()


    main(args)
