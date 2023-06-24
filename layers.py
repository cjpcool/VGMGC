import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn


class LatentMappingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, input_droprate=0., hidden_droprate=0.):
        super(LatentMappingLayer, self).__init__()
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, output_dim)
        self.input_droprate=input_droprate
        self.hidden_droprate = hidden_droprate

        # self._reset_parameters()

    def _reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.enc1.weight.size(1))
        stdv2 = 1. / math.sqrt(self.enc2.weight.size(1))
        # self.weight.data.normal_(-stdv, stdv)
        nn.init.kaiming_normal_(self.enc1.weight.data)
        nn.init.kaiming_normal_(self.enc2.weight.data)
        if self.enc1.bias is not None:
            self.enc1.bias.data.normal_(-stdv1, stdv1)

        if self.enc2.bias is not None:
            self.enc2.bias.data.normal_(-stdv2, stdv2)
            # nn.init.xavier_uniform_(self.bias.data, gain=1.414)

    def forward(self, x):
        z = self.encode(x)
        return z

    def encode(self, x):
        x = F.dropout(x, self.input_droprate, training=self.training)
        h = self.enc1(x)
        h = F.relu(h)
        h = F.dropout(h, self.hidden_droprate, training=self.training)
        h = self.enc2(h)
        return h


class GraphEncoder(nn.Module):
    def __init__(self, feat_dim, shared_feat_dim, hidden_dim, latent_dim, lam_emd=1., order=4):
        super(GraphEncoder, self).__init__()
        self.order = order
        self.lam_emd = lam_emd
        self.LatentMap = LatentMappingLayer(2 * feat_dim, hidden_dim, latent_dim)

        self.la = torch.ones(self.order)
        # self.la[1] = 2

    def forward(self, x, global_x, adj, global_adj, order=None):
        if order is not None:
            self.order = order
        e_global = self.message_passing_global(x, global_adj)
        # e_global = torch.matmul(global_adj, global_x)
        e_prim = self.message_passing_global(x, adj)
        e = torch.cat([e_prim, self.lam_emd * e_global], dim=-1)
        z = self.LatentMap(e)
        #################33
        # e_prim = self.message_passing_global(x, adj)
        # z = self.LatentMap(e_prim)
        return z

    def message_passing_global(self, x, adj):
        h = x
        for i in range(self.order):
            h = torch.matmul(adj, h) + (1 * x)

        return h

    def message_passing(self, x, adj):
        if self.order != 0:
            adj_temp = self.la[0] * adj.clone().detach_()
            adj_temp = adj_temp.to_dense()
            for i in range(self.order-1):
                adj_temp += self.la[i+1] * torch.spmm(adj, adj_temp).detach_() # matmul
            attn = adj_temp / self.order
            h2 = torch.mm(attn, x)

        else:
            h2 = x

        return h2


class MLPLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.normal_(-stdv, stdv)
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)
            # nn.init.xavier_uniform_(self.bias.data, gain=1.414)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, is_cuda=True, use_bn=False):
        super(MLP, self).__init__()

        self.layer1 = MLPLayer(nfeat, nhid)
        self.layer2 = MLPLayer(nhid, nclass)

        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.is_cuda = is_cuda
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn

    def forward(self, x):

        if self.use_bn:
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        x = F.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.layer2(x)

        return x

