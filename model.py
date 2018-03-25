import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class LBL(nn.Module):
    def __init__(self, pretrained_embeds, context_size, dropout=0.):
        super(LBL, self).__init__()
        # n in the paper
        self.context_size = context_size
        self.hidden_size = pretrained_embeds.size(1)
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
        norms = torch.norm(self.embedding_layer.weight, p=2, dim=1)
        #filter out vals where norm > max norm
        to_rescale = Variable(torch.from_numpy(
                np.where(norms.data.cpu().numpy() > max_norm)[0]))
        norms = torch.norm(self.embedding_layer(to_rescale), p=2, dim=1).data
        scaled = self.embedding_layer(to_rescale).div(
                Variable(norms.view(len(to_rescale), 1).expand_as(
                        self.embedding_layer(to_rescale)))).data
        self.embedding_layer.weight.data[to_rescale.long().data] = scaled


    def forward(self, context_words, gpu=True):
        self.batch_size = context_words.size(0)
        assert context_words.size(1) == self.context_size, \
            "context_words.size()=%s | context_size=%d" % \
            (context_words.size(), self.context_size)

        embeddings = self.embedding_layer(context_words)
        # sanity check
        assert embeddings.size() == \
            (self.batch_size, self.context_size, self.hidden_size)
        context_vectors = self.context_layer(embeddings.view(
                self.batch_size, self.context_size * self.hidden_size))
        context_vectors = self.dropout(context_vectors)
        assert context_vectors.size() == (self.batch_size, self.hidden_size)
        raw_outputs = self.output_layer(context_vectors)
        assert raw_outputs.size() == (self.batch_size, self.vocab_size)
        outputs = F.log_softmax(raw_outputs, dim=1)
        assert outputs.size() == (self.batch_size, self.vocab_size)
        return outputs


#Conditional Copy Model
class CopyProb(nn.Module):
    def __init__(self, pretrained_embeds, context_size, dropout=0.):
        super(CopyProb, self).__init__()
        self.context_size = context_size
        self.hidden_size = pretrained_embeds.size(1)
        self.vocab_size = pretrained_embeds.size(0)
        self.vocab = pretrained_embeds

        self.switch =\
            nn.Linear(self.hidden_size, 1)


    def get_train_parameters(self):
        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)
        return params


    def forward(self, context_vectors):
        switch = (F.sigmoid(self.switch(context_vectors)))
        switch = sum(switch)/len(switch)

        return switch


class CondCopy(nn.Module):
    def __init__(self, pretrained_embeds, context_size, dropout=0.):
        super(CondCopy, self).__init__()
        # n in the paper
        self.context_size = context_size
        self.hidden_size = pretrained_embeds.size(1)
        self.vocab_size = pretrained_embeds.size(0)
        
        #nn.Embedding(num embeddings, embedding dim)
        self.embedding_layer = nn.Embedding(
                self.vocab_size, self.hidden_size, max_norm=1, norm_type=2)
        self.max_norm_embedding()
        # C in the paper // nn.Linear (in features, out features) *doesn't learn additive bias
        self.context_layer = nn.Linear(
                self.hidden_size * int(self.context_size/10),
                self.hidden_size, bias=False)

        self.context_layer2 = nn.Linear(
                self.hidden_size* int(self.context_size/10), 
                self.hidden_size, bias=False)
        # dot product + bias in the paper
        self.output_shortlist =\
            nn.Linear(self.hidden_size, self.vocab_size) #affine1

        #output for location
        self.output_location =\
             nn.Linear(self.hidden_size, self.hidden_size) #affine2

        #switch probability
        self.copy =\
            nn.Linear(self.hidden_size, 1)
            #CopyProb(pretrained_embeds, context_size, dropout=0.)

        self.dropout = nn.Dropout(p=dropout)

        #self.copy_vec = Variable(torch.zeros(self.hidden_size, 1), requires_grad=True)

    def get_train_parameters(self):
        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def max_norm_embedding(self, max_norm=1):
        norms = torch.norm(self.embedding_layer.weight, p=2, dim=1)
        #filter out vals where norm > max norm
        to_rescale = Variable(torch.from_numpy(
                np.where(norms.data.cpu().numpy() > max_norm)[0]))
        norms = torch.norm(self.embedding_layer(to_rescale), p=2, dim=1).data
        scaled = self.embedding_layer(to_rescale).div(
                Variable(norms.view(len(to_rescale), 1).expand_as(
                        self.embedding_layer(to_rescale)))).data
        self.embedding_layer.weight.data[to_rescale.long().data] = scaled

    
    def reset_hidden(self):
        self.cvecs = None

    def forward(self, context_words):
        self.batch_size = context_words.size(0)
        assert context_words.size(1) == int(self.context_size/10), \
            "context_words.size()=%s | context_size=%d" % \
            (context_words.size(), self.context_size)

        #embedding layer
        embeddings = self.embedding_layer(context_words)
        # sanity check
        assert embeddings.size() == \
            (self.batch_size, int(self.context_size/10), self.hidden_size)

        embeds2 = F.relu(embeddings)

        assert embeds2.size() == \
            (self.batch_size, int(self.context_size/10), self.hidden_size)

        #get context vectors
        context_vectors = self.context_layer(embeddings.view(
                self.batch_size, int(self.context_size/10) * self.hidden_size))
        context_vectors = self.dropout(context_vectors)
        assert context_vectors.size() == (self.batch_size, self.hidden_size)

        #context vectors for pointer
        context_vecs2 = self.context_layer2(embeds2.view(
                self.batch_size, int(self.context_size/10) * self.hidden_size))
        context_vecs2 = self.dropout(context_vecs2)
        assert context_vecs2.size() == (self.batch_size, self.hidden_size)
        
        #shortlist softmax
        shortlist_outputs = self.output_shortlist(context_vectors)
        assert shortlist_outputs.size() == (self.batch_size, self.vocab_size)
        s_outputs = F.softmax(shortlist_outputs, dim=1)
        assert s_outputs.size() == (self.batch_size, self.vocab_size)


        #switch network -- probabililty 
        switch = (F.sigmoid(self.copy(context_vectors)))
        switch = sum(switch)/len(switch)

        #location softmax

        l_cvecs = F.tanh(self.output_location(context_vecs2)) 

        l_cvecs = l_cvecs*switch

        assert l_cvecs.size() == (self.batch_size, self.hidden_size)


        #(5) Multiply the output of step (4) by the matrix formed from the 4 context word embeddings (you will likely want to use batch matrix multiply (bmm) 
            #to accomplish this), to get scores that are batch_size x 4. Then apply a softmax to get a distribution over these preceding words.

        location_outputs = torch.bmm(embeddings, l_cvecs.view(self.batch_size, self.hidden_size, 1).contiguous())

        assert location_outputs.size() == (self.batch_size, int(self.context_size/10), 1)

        location_outputs = torch.squeeze(location_outputs)

        assert location_outputs.size() == (self.batch_size, int(self.context_size/10))

        l_outputs = F.softmax(location_outputs, dim=1)

        assert l_outputs.size() == (self.batch_size, int(self.context_size/10))

        #6) Now you need to somehow combine the two distributions you formed in steps (3) and (5). I guess the easiest approach is to have another 
    #distribution that tells you whether you copied or not. Then, p(word5 | ctx) = p(copied) * pointer_probability_of_word5 + (1 - p(copied)) * probability_of_word5_from_step3.  
    #Note that pointer_probability_of_word5 might be zero. 

        #pre_mat = cumulate 

        output = torch.log(torch.cat((1-switch)*s_outputs, switch*l_outputs, dim=1))

        #compute pointer softmax
        #output = ((switch*l_outputs),  ((1-switch)*s_outputs))

        #return torch.log(l_outputs),  torch.log((1-switch)*s_outputs)

        return output


