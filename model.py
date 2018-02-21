import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class LBL(nn.Module):
    def __init__(self, pretrained_embeds, context_size, dropout=0.5):
        super(LBL, self).__init__()
        # n in the paper
        self.context_size = context_size
        self.hidden_size = 1500 #pretrained_embeds.size(1)
        self.vocab_size = pretrained_embeds.size(0)
		
		#nn.Embedding(num embeddings, embedding dim)
        self.embedding_layer = nn.Embedding(
                self.vocab_size, self.hidden_size)
        self.max_norm_embedding()
        # C in the paper // nn.Linear (in features, out features) *doesn't learn additive bias
        self.context_layer = nn.Linear(
                self.hidden_size * self.context_size,
                self.hidden_size, bias=False)
        # dot product + bias in the paper
        self.output_layer =\
            nn.Linear(self.hidden_size, self.vocab_size)

        self.dropout = nn.Dropout(p=dropout)

    def get_train_parameters(self):
        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)
        return params
	
	#dropout: During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a bernoulli distribution. The elements to zero are randomized on every forward call.
	#torch.norm(input tensor, p=2, dim) p = exponent val in norm formulation, dim = dimension to reduce
    #make sure weights never exceeds a certain threshold 
    def max_norm_embedding(self, max_norm=1):
        norms = torch.norm(self.embedding_layer.weight, p=2, dim=1).cuda()
        #filter out vals where norm > max norm
        to_rescale = Variable(torch.from_numpy(
                np.where(norms.data.cpu().numpy() > max_norm)[0])).cuda()
        norms = torch.norm(self.embedding_layer(to_rescale), p=2, dim=1).data.cuda()
        scaled = self.embedding_layer(to_rescale).div(
                Variable(norms.view(len(to_rescale), 1).expand_as(
                        self.embedding_layer(to_rescale)))).data.cuda()
        self.embedding_layer.weight.data[to_rescale.long().data] = scaled

    def forward(self, context_words):
        self.batch_size = context_words.size(0).cuda()
        assert context_words.size(1) == self.context_size, \
            "context_words.size()=%s | context_size=%d" % \
            (context_words.size(), self.context_size)

        embeddings = self.embedding_layer(context_words).cuda()
        # sanity check
        assert embeddings.size() == \
            (self.batch_size, self.context_size, self.hidden_size)
        context_vectors = self.context_layer(embeddings.view(
                self.batch_size, self.context_size * self.hidden_size)).cuda()
        context_vectors = self.dropout(context_vectors)
        assert context_vectors.size() == (self.batch_size, self.hidden_size)
        raw_outputs = self.output_layer(context_vectors)
        assert raw_outputs.size() == (self.batch_size, self.vocab_size)
        outputs = F.log_softmax(raw_outputs)
        assert outputs.size() == (self.batch_size, self.vocab_size)
        return outputs


#Conditional Copy Model

class CondCopy(nn.Module):
    def __init__(self, pretrained_embeds, context_size, dropout=0.5):
        super(CondCopy, self).__init__()
        # n in the paper
        self.context_size = context_size
        self.hidden_size = 1500 #pretrained_embeds.size(1) 
        self.vocab_size = pretrained_embeds.size(0)
        
        #nn.Embedding(num embeddings, embedding dim)
        self.embedding_layer = nn.Embedding(
                self.vocab_size, self.hidden_size)
        self.max_norm_embedding()
        # C in the paper // nn.Linear (in features, out features) *doesn't learn additive bias
        self.context_layer = nn.Linear(
                self.hidden_size * self.context_size,
                self.hidden_size, bias=False)
        # dot product + bias in the paper
        self.output_shortlist =\
            nn.Linear(self.hidden_size, self.vocab_size)

        self.output_location =\
            nn.Linear(self.hidden_size, self.hidden_size) #or context size

        self.switch =\
            nn.Linear(self.hidden_size, 1)

        self.dropout = nn.Dropout(p=dropout)

    def get_train_parameters(self):
        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)
        return params
    
    #dropout: During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a bernoulli distribution. The elements to zero are randomized on every forward call.
    #torch.norm(input tensor, p=2, dim) p = exponent val in norm formulation, dim = dimension to reduce
    #make sure weights never exceeds a certain threshold 
    def max_norm_embedding(self, max_norm=1):
        norms = torch.norm(self.embedding_layer.weight, p=2, dim=1).cuda()
        #filter out vals where norm > max norm
        to_rescale = Variable(torch.from_numpy(
                np.where(norms.data.cpu().numpy() > max_norm)[0])).cuda()
        norms = torch.norm(self.embedding_layer(to_rescale), p=2, dim=1).data.cuda()
        scaled = self.embedding_layer(to_rescale).div(
                Variable(norms.view(len(to_rescale), 1).expand_as(
                        self.embedding_layer(to_rescale)))).data.cuda()
        self.embedding_layer.weight.data[to_rescale.long().data] = scaled

    def pointer_softmax(self, shortlist, location, switch_net):
        p_short = shortlist * switch_net
        p_loc = location * (1 - switch_net)
        return torch.cat((p_short, p_loc), 1)

    def forward(self, context_words):
        self.batch_size = context_words.size(0).cuda()
        assert context_words.size(1) == self.context_size, \
            "context_words.size()=%s | context_size=%d" % \
            (context_words.size(), self.context_size)

        #embedding layer
        embeddings = self.embedding_layer(context_words).cuda()
        # sanity check
        assert embeddings.size() == \
            (self.batch_size, self.context_size, self.hidden_size)
        
        #get context vectors
        context_vectors = self.context_layer(embeddings.view(
                self.batch_size, self.context_size * self.hidden_size)).cuda()
        context_vectors = self.dropout(context_vectors)
        assert context_vectors.size() == (self.batch_size, self.hidden_size)
        
        #shortlist softmax
        shortlist_outputs = self.output_shortlist(context_vectors)
        assert shortlist_outputs.size() == (self.batch_size, self.vocab_size)
        s_outputs = F.log_softmax(shortlist_outputs)
        assert s_outputs.size() == (self.batch_size, self.vocab_size)

        #location softmax
        location_outputs = self.output_location(context_vectors)
        l_outputs = F.log_softmax(location_outputs)

        #switch network -- probabililty 
        switch = F.sigmoid(self.switch(context_vectors))

        #compute pointer softmax
        output = self.pointer_softmax(s_outputs, l_outputs, switch)

        return output
