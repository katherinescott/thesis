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
        self.vocab = pretrained_embeds
        
        #nn.Embedding(num embeddings, embedding dim)
        self.embedding_layer = nn.Embedding(
                self.vocab_size, self.hidden_size, max_norm=1, norm_type=2)
        self.max_norm_embedding()
        # C in the paper // nn.Linear (in features, out features) *doesn't learn additive bias
        self.context_layer = nn.Linear(
                self.hidden_size * int(self.context_size/10),
                self.hidden_size, bias=False)

        self.context_layer2 = nn.Linear(
                self.hidden_size, self.hidden_size, bias=False)
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

    def forward(self, context_words):
        self.batch_size = context_words.size(0)
        assert context_words.size(1) == int(self.context_size/10), \
            "context_words.size()=%s | context_size=%d" % \
            (context_words.size(), self.context_size)

        length = context_words.size(1)

        probs = []
        hiddens = []

        cumulate = torch.zeros((length, self.batch_size, self.vocab_size))
        cumulate.scatter_(2, context_words.transpose(0,1).data.type(torch.LongTensor).unsqueeze(2), 1.0)

        point_scores = []
        
        for i in range(length):
            embeddings = self.embedding_layer(context_words.transpose(0,1)[i]) #now its contxt x batch
            #assert embeddings.size() == \
                #(int(self.context_size/10), self.batch_size, self.hidden_size)

            cvecs = self.context_layer2(embeddings) #.view(self.batch_size, int(self.context_size/10) * self.hidden_size))
            
            hiddens.append(cvecs)

            q = F.tanh(self.output_location(cvecs))

            #switch probability
            switch = F.sigmoid(self.copy(cvecs))
            switch = sum(switch)/len(switch)

            z = []
            for j in range(i+1):
                z.append(torch.sum(hiddens[j]*q, 1).view(-1))
            z.append(torch.mul(q, switch).view(-1))
            #z = torch.stack(z)

            a = F.softmax(z.transpose(0,1))
            prefix_matrix = cumulate_matrix[:i + 1]
            p_ptr = torch.sum(Variable(prefix_matrix) * a[:-1].unsqueeze(2).expand_as(prefix_matrix), 0).squeeze(0)

            out = self.output_shortlist(cvecs)
            prob_vocab = F.softmax(out)

            p = p_ptr + p_vocab * a[-1].unsqueeze(1).expand_as(p_vocab)

            probs.append(p)

        return torch.log(torch.cat(probs).view(-1, self.vocab_size)), torch.log(torch.cat(ptr_scores).view(-1, self.vocab_size))



