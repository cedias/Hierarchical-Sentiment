import argparse
import pickle as pkl
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Nets import NSCUPA, HAN
from fmtl import FMTL
from utils import *


def save(net, dic, path):
    dict_m = net.state_dict()
    dict_m["word_dic"] = dic    
    torch.save(dict_m,path)

def tuple_batch(l):
    user, item, review,rating = zip(*l)
    r_t = torch.Tensor(rating).long()
    u_t = torch.Tensor(user).long()
    i_t = torch.Tensor(item).long()
    list_rev = review

    sorted_r = sorted([(len(r),r_n,r) for r_n,r in enumerate(list_rev) ],reverse=True) #index by desc rev_le
    lr, r_n, ordered_list_rev = zip(*sorted_r)
    
    lr = list(lr)

    max_sents = lr[0]
    #reordered
    r_t = r_t[[r_n]]
    u_t = u_t[[r_n]]
    i_t = i_t[[r_n]]
    review = [review[x] for x in r_n] #reorder reviews

    stat =  sorted([(len(s), r_n, s_n, s) for r_n, r in enumerate(ordered_list_rev) for s_n, s in enumerate(r)], reverse=True)
    max_words = stat[0][0]

    ls = []
    batch_t = torch.zeros(len(stat), max_words).long()                          # (sents ordered by len)
    ui_indexs = torch.zeros(len(stat)).long()                                  # (sents original rev_n)
    sent_order = torch.zeros(len(ordered_list_rev), max_sents).long().fill_(0) # (rev_n,sent_n)

    for i,s in enumerate(stat):
        sent_order[s[1],s[2]] = i+1
        ui_indexs[i]=s[1]
        batch_t[i,0:len(s[3])] = torch.LongTensor(s[3])
        ls.append(s[0])

    return batch_t,r_t,u_t,i_t,sent_order,ui_indexs,ls,lr,review



def train(epoch,net,dataset,device,msg="val/test",optimize=False,optimizer=None,criterion=None):

    if optimize:
        net.train()
    else:
        net.eval()

    epoch_loss = 0
    mean_mse = 0
    mean_rmse = 0
    ok_all = 0
    #data_tensors = new_tensors(3,cuda,types={0:torch.LongTensor,1:torch.LongTensor,2:torch.LongTensor}) #data-tensors

    with tqdm(total=len(dataset),desc=msg) as pbar:
        for iteration, (batch_t,r_t,u_t,i_t,sent_order,ui_indexs,ls,lr,review) in enumerate(dataset):

            data = (batch_t,r_t,u_t,i_t,sent_order,ui_indexs)
            data = list(map(lambda x:x.to(device),data))

            if optimize:
                optimizer.zero_grad()

           
            out = net(data[0],data[2],data[3],data[4],data[5],ls,lr)

            ok,per,val_i = accuracy(out,data[1])
            ok_all += per.item()

            mseloss = F.mse_loss(val_i,data[1].float())
            mean_rmse += math.sqrt(mseloss.item())
            mean_mse += mseloss.item()

            if optimize:
                loss =  criterion(out, data[1]) 
                epoch_loss += loss.item()
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
    data_tensors = new_tensors(6,cuda,types={0:torch.LongTensor,1:torch.LongTensor,2:torch.LongTensor,3:torch.LongTensor,4:torch.LongTensor,5:torch.LongTensor}) #data-tensors
    
    with tqdm(total=len(dataset),desc=msg) as pbar:
        for iteration, (batch_t,r_t,u_t,i_t,sent_order,ui_indexs,ls,lr,review) in enumerate(dataset):
            data = tuple2var(data_tensors,(batch_t,r_t,u_t,i_t,sent_order,ui_indexs))
            out  = net(data[0],data[2],data[3],data[4],data[5],ls,lr)
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

    datadict = pkl.load(open(args.filename,"rb"))
    data_tl,(trainit,valit,testit) = FMTL_train_val_test(datadict["data"],datadict["splits"],args.split,validation=0.5,rows=datadict["rows"])

    rating_mapping = data_tl.get_field_dict("rating",key_iter=trainit) #creates class mapping
    data_tl.set_mapping("rating",rating_mapping) 

    user_mapping = data_tl.get_field_dict("user_id",key_iter=trainit,offset=1) #creates class mapping
    data_tl.set_mapping("user_id",user_mapping,unk=0) # if unknown #id is 0
    user_mapping["_unk_"] = 0

    item_mapping = data_tl.get_field_dict("item_id",key_iter=trainit,offset=1) #creates class mapping
    data_tl.set_mapping("item_id",item_mapping,unk=0)
    item_mapping["_unk_"] = 0

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
        net = NSCUPA(ntoken=len(state["word_dic"]),nusers=state["users.weight"].size(0), nitems=state["items.weight"].size(0),emb_size=state["embed.weight"].size(1),hid_size=state["sent.rnn.weight_hh_l0"].size(1),num_class=state["lin_out.weight"].size(0))
        del state["word_dic"]
        net.load_state_dict(state)

    else:
        if args.emb:
            net = NSCUPA(ntoken=len(wdict),nusers=len(user_mapping), nitems=len(item_mapping),emb_size=len(tensor[1]),hid_size=args.hid_size,num_class=len(rating_mapping))
            net.set_emb_tensor(torch.FloatTensor(tensor))
        else:
            net = NSCUPA(ntoken=len(wdict),nusers=len(user_mapping), nitems=len(item_mapping), emb_size=args.emb_size,hid_size=args.hid_size, num_class=len(rating_mapping))

    if args.prebuild:
        data_tl = FMTL(list(x for x  in tqdm(data_tl,desc="prebuilding")),data_tl.rows)

    return data_tl,(trainit,valit,testit), net, wdict


def main(args):

    print(32*"-"+"\nNeural Sentiment Classification with User & Product Attention:\n" + 32*"-")
    data_tl, (train_set, val_set, test_set), net, wdict = load(args)


    dataloader = DataLoader(data_tl.indexed_iter(train_set), batch_size=args.b_size, shuffle=True, num_workers=3, collate_fn=tuple_batch,pin_memory=True)
    dataloader_valid = DataLoader(data_tl.indexed_iter(val_set), batch_size=args.b_size, shuffle=False,  num_workers=3, collate_fn=tuple_batch)
    dataloader_test = DataLoader(data_tl.indexed_iter(test_set), batch_size=args.b_size, shuffle=False, num_workers=3, collate_fn=tuple_batch,drop_last=True)

    criterion = torch.nn.CrossEntropyLoss()      

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.cuda:
        net.to(device)

    print("-"*20)

    optimizer = optim.Adam(net.parameters())
    torch.nn.utils.clip_grad_norm(net.parameters(), args.clip_grad)

    for epoch in range(1, args.epochs + 1):
        print("\n-------EPOCH {}-------".format(epoch))
        train(epoch,net,dataloader,device,msg="training",optimize=True,optimizer=optimizer,criterion=criterion)

        if args.snapshot:
            print("snapshot of model saved as {}".format(args.save+"_snapshot"))
            save(net,wdict,args.save+"_snapshot")

        train(epoch,net,dataloader_valid,device,msg="Validation")
        train(epoch,net,dataloader_test,device,msg="Evaluation")

    if args.save:
        print("model saved to {}".format(args.save))
        save(net,wdict,args.save)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Neural Sentiment Classification with User & Product Attention')
    
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

    main(args)
