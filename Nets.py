import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import sys
from operator import itemgetter
from torch.autograd import Variable
from collections import OrderedDict


class AttentionalBiRNN(nn.Module):

    def __init__(self, inp_size, hid_size, dropout=0, RNN_cell=nn.GRU):
        super(AttentionalBiRNN, self).__init__()
        self.register_buffer("mask",torch.FloatTensor())

        natt = hid_size*2
        self.gru = RNN_cell(input_size=inp_size,hidden_size=hid_size,num_layers=1,bias=True,batch_first=True,dropout=dropout,bidirectional=True)
        self.att_w = nn.Linear(natt,1,bias=False) # v transpose is here

        self.att_u = nn.Linear(inp_size,natt,bias=False)
        self.att_h = nn.Linear(natt,natt,bias = False)
        self.att_i = nn.Linear(inp_size,natt) # holds the bias of tanh(w_h*h+w_u*u+w_i*i + b)
        

    #
    def forward(self, packed_batch,user_embs,item_embs):
        
        rnn_sents,_ = self.gru(packed_batch)
        enc_sents,len_s = torch.nn.utils.rnn.pad_packed_sequence(rnn_sents)

        sum_ue = self.att_u(user_embs) + self.att_i(item_embs)

        transformed_h = self.att_h(enc_sents.view(enc_sents.size(0)*enc_sents.size(1),-1))
        summed = F.tanh(sum_ue + transformed_h.view(enc_sents.size()))
        att = self.att_w(summed.view(summed.size(0)*summed.size(1),-1)).view(summed.size(0),summed.size(1)).transpose(0,1)
        all_att = self._masked_softmax(att,self._list_to_bytemask(list(len_s))).transpose(0,1) # attW,sent 
        attended = all_att.unsqueeze(-1) * enc_sents
        return attended.sum(0,True).squeeze(0)

    def forward_att(self, packed_batch):
        
        rnn_sents,_ = self.gru(packed_batch)
        enc_sents,len_s = torch.nn.utils.rnn.pad_packed_sequence(rnn_sents)
        
        emb_h = self.tanh(self.lin(enc_sents.view(enc_sents.size(0)*enc_sents.size(1),-1)))  # Nwords * Emb
        attend = self.att_w(emb_h).view(enc_sents.size(0),enc_sents.size(1)).transpose(0,1)
        all_att = self._masked_softmax(attend,self._list_to_bytemask(list(len_s))).transpose(0,1) # attW,sent 
        attended = all_att.unsqueeze(2).expand_as(enc_sents) * enc_sents
        return attended.sum(0,True).squeeze(0), all_att

    #
    def _list_to_bytemask(self,l):
        mask = self._buffers['mask'].resize_(len(l),l[0]).fill_(1)

        for i,j in enumerate(l):
            if j != l[0]:
                mask[i,j:l[0]] = 0

        return mask
    
    #
    def _masked_softmax(self,mat,mask):

        exp = torch.exp(mat) * Variable(mask,requires_grad=False)
        sum_exp = exp.sum(1,True)+0.0001
     
        return exp/sum_exp.expand_as(exp)


class HierarchicalDoc(nn.Module):

    def __init__(self, ntoken, nusers, nitems, num_class, emb_size=200, hid_size=50):
        super(HierarchicalDoc, self).__init__()

        self.embed = nn.Embedding(ntoken, emb_size, padding_idx=0)
        self.users = nn.Embedding(nusers, emb_size)
        I.normal(self.users.weight.data,0.01,0.01)
        self.items = nn.Embedding(nitems, emb_size)
        I.normal(self.items.weight.data,0.01,0.01)


        self.word = AttentionalBiRNN(emb_size, emb_size//2)
        self.sent = AttentionalBiRNN(emb_size, emb_size//2)

        self.emb_size = emb_size
        self.lin_out = nn.Linear(emb_size,num_class)
        self.register_buffer("reviews",torch.Tensor())

    def set_emb_tensor(self,emb_tensor):
        self.emb_size = emb_tensor.size(-1)
        self.embed.weight.data = emb_tensor

    #
    def _reorder_sent(self,sents,sent_order,lr):

        revs = Variable(self._buffers["reviews"].resize_(len(lr),lr[0],sents.size(1)).fill_(0), requires_grad=False)
        
        for i,len_rev in enumerate(lr):
            rev_s = sent_order[i,:len_rev]
            revs[i,0:len_rev,:] = sents[rev_s]

        return revs
        
    #
    def forward(self, batch_reviews,users,items,sent_order,ui_indexs,ls,lr):
        
        
        u = users[ui_indexs]
        i = items[ui_indexs]


        emb_w = F.dropout(self.embed(batch_reviews),training=self.training)
        emb_u = F.dropout(self.users(u),training=self.training)
        emb_i = F.dropout(self.items(i),training=self.training)
        
        packed_sents = torch.nn.utils.rnn.pack_padded_sequence(emb_w, ls,batch_first=True)
       

        sent_embs = self.word(packed_sents,emb_u,emb_i)
        rev_embs = self._reorder_sent(sent_embs,sent_order,lr)

        packed_rev = torch.nn.utils.rnn.pack_padded_sequence(rev_embs, lr,batch_first=True)

        emb_u = F.dropout(self.users(users),training=self.training)
        emb_i = F.dropout(self.items(items),training=self.training)

        doc_embs = self.sent(packed_rev,emb_u,emb_i)

        out = self.lin_out(doc_embs)

        return out


    def forward_visu(self, batch_reviews,stats):
        ls,lr,rn,sn = zip(*stats)
        emb_w = self.embed(batch_reviews)
        
        packed_sents = torch.nn.utils.rnn.pack_padded_sequence(emb_w, ls,batch_first=True)
        sent_embs,att_w = self.word.forward_att(packed_sents)
        
        rev_embs,lens,real_order = self._reorder_sent(sent_embs,zip(lr,rn,sn))

        packed_rev = torch.nn.utils.rnn.pack_padded_sequence(rev_embs, lens,batch_first=True)
        doc_embs,att_s = self.sent.forward_att(packed_rev)

        final_emb = doc_embs[real_order,:]
        att_s = att_s[:,real_order]

        out = self.lin_out(final_emb)
        
        return out,att_s




