import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False, incidence_v=100, incidence_e=50,
                 init_dist=None, args=None):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        self.args = args
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.adj = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l, G):
        self.adj = adj
        theta = math.log(lamda / l + 1)
        # theta=1
        hi = torch.spmm(G, input)
        # alpha=(math.log(l)-math.log(1))/(math.log(64)-math.log(1))*0.1+0.3
        # alpha=0
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class GCNII(nn.Module):

    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, incidence_v=100, incidence_e=50,
                 init_dist=None, args=None):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.in_features = nfeat
        self.out_features = nclass
        self.hiddendim = nhidden
        self.nhiddenlayer = nlayers

        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, input, adj, G):
        _layers = []
        x = F.dropout(input, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        # layer_inner = input
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1, G=G))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        self.adj = con.adj  # 保存看看学的结果
        return layer_inner  # F.log_softmax(layer_inner, dim=1)

    def get_outdim(self):
        return self.out_features

    def __repr__(self):
        return "%s lamda=%s alpha=%s (%d - [%d:%d] > %d)" % (self.__class__.__name__,self.lamda,
                                                    self.alpha,
                                                    self.in_features,
                                                    self.hiddendim,
                                                    self.nhiddenlayer,
                                                    self.out_features)
