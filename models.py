import warnings

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
from layers import GraphEncoder, LatentMappingLayer, MLP
import numpy as np
Tensor = torch.Tensor


class GraphGenerator(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim, input_droprate=0.1, hidden_droprate=0.1, temp=0.2, threshold=0.5):
        super(GraphGenerator, self).__init__()
        self.temp =temp
        self.threshold = threshold
        self.encoder = MLP(feat_dim, hidden_dim, latent_dim, input_droprate=input_droprate, hidden_droprate=hidden_droprate)
        self.decoder = MLP(latent_dim, hidden_dim, feat_dim, input_droprate=input_droprate, hidden_droprate=hidden_droprate)
        # self.bn = nn.BatchNorm1d(latent_dim, affine=False, track_running_stats=False)
        self.W = nn.Parameter(torch.FloatTensor(latent_dim,latent_dim))
        nn.init.xavier_normal_(self.W.data)

    def forward(self, x):
        h = self.encoder(x)

        h_norm = F.normalize(h, p=2, dim=1)
        h = F.relu(h)

        K = torch.matmul(h, self.W)
        attn = torch.matmul(K, h.t())

        attn = F.normalize(attn, p=2, dim=1)

        x_pred = self.decoder(K)
        x_pred = torch.sigmoid(x_pred)
        return attn, x_pred, h_norm


    def reparameter(self, attn, hard, training=False):
        attn = GraphGenerator.concrete_sigmoid(attn, tau=self.temp, hard=hard, threshold=self.threshold, training=training)
        if hard:
            attn += torch.eye(attn.shape[0], device=attn.device)
            attn = GraphGenerator.normalize_adj(attn)

        return attn

    @staticmethod
    def samplel_discrete_uniform(logits, eps=1e-10):
        U = torch.rand_like(logits, memory_format=torch.legacy_contiguous_format, device=logits.device)
        return torch.log(U + eps) - torch.log(1 - U + eps)
        # return U.exponential_().log()

    @staticmethod
    def concrete_sigmoid(logits: Tensor, tau: float = .5, hard: bool = False, eps: float = 1e-10, threshold=0.5, training=True,
                         dim: int = -1) -> Tensor:
        # if has_torch_function_unary(logits):
        #     return handle_torch_function(gumbel_softmax, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim)
        if eps != 1e-10:
            warnings.warn("`eps` parameter is deprecated and has no effect.")

        # gumbels = (
        #     -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        # )  # ~Gumbel(0,1)
        # gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        if training:
            g = GraphGenerator.samplel_discrete_uniform(logits, eps=eps)
            g = (g + logits) / tau
        else:
            g = logits
        y_soft = g.sigmoid()

        if hard:
            # Straight through.
            # index = y_soft.max(dim, keepdim=True)[1]
            # y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
            y_hard[y_soft > threshold] = 1.
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

    @staticmethod
    def normalize_adj(x):
        # laplacian normalization
        # rowsum = x.sum(1).detach().clone()
        # colsum = x.sum(0).detach().clone()
        #
        # r_inv = rowsum.pow(-0.5).flatten()
        #
        # r_inv[torch.isinf(r_inv)] = 0.
        # c_inv = colsum.pow(-0.5).flatten()
        # c_inv[torch.isinf(c_inv)] = 0.
        # r_inv = r_inv.reshape((x.shape[0], -1))
        # c_inv = c_inv.reshape((-1, x.shape[1]))
        # x = x * r_inv * c_inv

        # row normalization
        D = x.sum(1).detach().clone()
        r_inv = D.pow(-1).flatten()
        r_inv = r_inv.reshape((x.shape[0], -1))
        r_inv[torch.isinf(r_inv)] = 0.
        x = x * r_inv

        # x = F.normalize(x, p=1, dim=-1)
        return x


class MultiGraphAutoEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim, class_num, lam_emd=1., alpha=0.2, order=5, view_num=1, temperature=0.3,threshold=0.7):
        super(MultiGraphAutoEncoder, self).__init__()
        self.hidden_size = hidden_dim
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.class_num = class_num
        self.lam_emd = lam_emd
        self.threshold=threshold

        self.cluster_layer = [Parameter(torch.Tensor(class_num, latent_dim)) for _ in range(view_num)]
        self.cluster_layer.append(Parameter(torch.Tensor(class_num, view_num * latent_dim)))
        # self.cluster_layer.append(torch.cat(self.cluster_layer, dim=-1))

        self.GraphEnc = [GraphEncoder(feat_dim[i], feat_dim[-1], hidden_dim, latent_dim, lam_emd=lam_emd, order=order) for i in range(view_num)]
        # self.LatentMap = [LatentMappingLayer(2*feat_dim, hidden_dim, latent_dim) for _ in range(view_num)]
        self.FeatDec = [LatentMappingLayer(latent_dim, hidden_dim, feat_dim[i]) for i in range(view_num)]

        for i in range(view_num):
            self.register_parameter('centroid_{}'.format(i), self.cluster_layer[i])
            self.add_module('graphenc_{}'.format(i), self.GraphEnc[i])
            # self.add_module('latentmap_{}'.format(i), self.LatentMap[i])
            self.add_module('featdec_{}'.format(i), self.FeatDec[i])
        self.register_parameter('centroid_{}'.format(view_num), self.cluster_layer[view_num])

        self.GraphGen = GraphGenerator(feat_dim[-1], 512, 512, input_droprate=0.2, hidden_droprate=0.2, temp=temperature, threshold=threshold)

    def forward(self, x, shared_x, adj, view=0):
        attn, x_global_pred, h = self.GraphGen(shared_x)
        A_global = self.GraphGen.reparameter(attn, hard=True, training=False)
        z = self.GraphEnc[view](x, shared_x, adj, A_global)

        z_norm = F.normalize(z, p=2, dim=1)
        q = self.predict_distribution(z_norm, view)

        if self.training:
            A_pred = self.decode(z_norm)
            x_prim = self.FeatDec[view](F.relu(z))
            x_pred = torch.sigmoid(x_prim)
            return A_pred, z_norm, q, x_pred, x_global_pred, attn
        else:
            return q, z_norm

    # @staticmethod
    def decode(self, z):
        rec_graph = torch.sigmoid(torch.matmul(z, z.t()))
        return rec_graph

    def predict_distribution(self, z, v, alpha=1.0):
        c = self.cluster_layer[v]
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - c, 2), 2) / alpha)
        q = q.pow((alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def get_graph_embedding(self, x, adj,view):
        e = self.GraphEnc[view](x, adj)
        e_norm = F.normalize(e, p=2, dim=1)

        return e_norm




