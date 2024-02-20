import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MultiLayerHGCN(nn.Module):
    """
        The base block for Multi-layer GCN / ResGCN / Dense GCN
        """

    def __init__(self, in_features, hidden_features, out_dim, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=0.5,
                 aggrmethod="nores", dense=False, res=False, incidence_v=None, incidence_e=None,
                 args=None):
        """
        The base block for constructing DeepGCN model.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param dense: enable dense connection
        """
        super(MultiLayerHGCN, self).__init__()
        self.in_features = in_features
        self.hiddendim = hidden_features
        self.nhiddenlayer = nbaselayer
        self.out_dim = out_dim
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dense = dense
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop
        self.hiddenlayers = nn.ModuleList()
        self.baselayer = HGraphConvolutionBS
        self.res = res
        self.incidence_v = incidence_v
        self.incidence_e = incidence_e


        self.args = args
        self.__makehidden()
        self.adj = None

    def __makehidden(self):
        for i in range(self.nhiddenlayer):
            if i == 0:
                layer = self.baselayer(self.in_features, self.hiddendim, activation=self.activation, withbn=self.withbn,
                                       res=self.res,
                                        args=self.args)
            elif i==self.nhiddenlayer-1:
                layer = self.baselayer(self.hiddendim, self.out_dim, activation=lambda x:x, withbn=self.withbn,
                                       res=self.res,
                                       args=self.args)
            else:
                layer = self.baselayer(self.hiddendim, self.hiddendim, activation=self.activation, res=self.res,
                                        withbn=self.withbn
                                       , args=self.args)

            self.hiddenlayers.append(layer)

    def _doconcat(self, x, subx):
        if x is None:
            return subx
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx
        elif self.aggrmethod == "nores":
            return x

    def forward(self, input, adj=None, G=None):
        x = input
        denseout = None
        # Here out is the result in all levels.
        for num, gc in enumerate(self.hiddenlayers):
            denseout = self._doconcat(denseout, x)
            x = gc(input=x, adj=adj, G=G)
            if num<self.nhiddenlayer-1:
                x = F.dropout(x, self.dropout, training=self.training)

        if not self.dense:
            return self._doconcat(x, input)
        return self._doconcat(x, denseout)

    def get_outdim(self):
        return self.hiddendim


class HGraphConvolutionBS(nn.Module):

    """
    HGCN Layer with BN, Self-loop and Res connection.
    """

    def __init__(self, in_features, out_features, activation=lambda x: x, withbn=False, withloop=False, bias=True,
                 res=False,args=None, ):
        """
        Initial function.
        :param in_features: the input feature dimension.
        :param out_features: the output feature dimension.
        :param activation: the activation function.
        :param withbn: using batch normalization.
        :param withloop: using self feature modeling.
        :param bias: enable bias.
        :param res: enable res connections.
        """
        super(HGraphConvolutionBS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = activation
        self.res = res
        self.args = args

        # Parameter setting.
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if withloop:
            self.self_weight = Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.register_parameter("self_weight", None)

        if withbn:
            self.bn = torch.nn.BatchNorm1d(out_features)
        else:
            self.register_parameter("bn", None)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        # self.dynamic_adj = Flgc2d(e_n=incidence_e, v_n=incidence_v, init_dist=init_dist, only_G=True)
        self.K_neigs = args.K_neigs
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.self_weight is not None:
            stdv = 1. / math.sqrt(self.self_weight.size(1))
            self.self_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj=None, G=None):

        support = torch.matmul(input, self.weight)
        if self.bias is not None:
            support = support + self.bias
        output = torch.spmm(G, support)

        # Self-loop
        if self.self_weight is not None:
            output = output + torch.mm(input, self.self_weight)

        if self.bn is not None:
            output = self.bn(output)
        # Res
        if self.res:
            return self.sigma(output) + input
        else:
            return self.sigma(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Dense(nn.Module):
    """
    Simple Dense layer, Do not consider adj.
    """

    def __init__(self, in_features, out_features, activation=lambda x: x, withbn=False, bias=True, res=False):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = activation
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.res = res
        self.adj = None  # 为了验证adj 的有效性设计的（在GCMmodel里边�?
        # self.bn = nn.BatchNorm1d(out_features)

        if withbn:
            self.bn = torch.nn.BatchNorm1d(out_features)
        else:
            self.register_parameter("bn", None)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj=None, G=None):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            output = output + self.bias
        if self.bn is not None:
            output = self.bn(output)
        # output = self.bn(output)
        return self.sigma(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

