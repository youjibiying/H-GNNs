from torch import nn
import torch
from models.layers import Dense, MultiLayerHGCN
from models.mlp import MLP
from models.gcnii import GCNII


# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class GCNModel(nn.Module):
    """
       The model architecture likes:
       All options are configurable.
    """

    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 nhidlayer,
                 dropout,
                 baseblock="mutigcn",
                 inputlayer=None,
                 outputlayer=None,
                 nbaselayer=0,
                 activation=lambda x: x,
                 withbn=False,
                 withloop=False,
                 aggrmethod="add",
                 mixmode=False,
                 args=None,
                 ):
        """
        Initial function.
        :param nfeat: the input feature dimension.
        :param nhid:  the hidden feature dimension.
        :param nclass: the output feature dimension.
        :param nhidlayer: the number of hidden blocks.
        :param dropout:  the dropout ratio.
        :param baseblock: the baseblock type, can be "mutigcn", "resgcn", "densegcn" and "inceptiongcn".
        :param inputlayer: the input layer type, can be "gcn", "dense", "none".
        :param outputlayer: the input layer type, can be "gcn", "dense".
        :param nbaselayer: the number of layers in one hidden block.
        :param activation: the activation function, default is ReLu.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param mixmode: enable cpu-gpu mix mode. If true, put the inputlayer to cpu.
        """
        super(GCNModel, self).__init__()
        self.dropout = dropout
        self.baseblock = baseblock.lower()
        self.nbaselayer = nbaselayer
        self.inputlayer = inputlayer
        self.outputlayer = outputlayer
        self.args = args

        if baseblock == "gcnii":
            self.BASEBLOCK = GCNII
        elif self.baseblock == "gcn":
            self.BASEBLOCK = MultiLayerHGCN
        # elif self.baseblock == 'shsc':
        #     self.BASEBLOCK = SHSC
        elif self.baseblock == "mlp":
            self.BASEBLOCK = MLP
        else:
            raise NotImplementedError("Current baseblock %s is not supported." % (baseblock))

        self.midlayer = nn.ModuleList()

        for i in range(nhidlayer):
            if baseblock.lower() == 'gcnii':
                gcb = self.BASEBLOCK(nfeat=nfeat,
                                     nlayers=args.degree,
                                     nhidden=nhid,
                                     nclass=nclass,
                                     dropout=dropout,
                                     lamda=args.lamda,
                                     alpha=args.alpha,
                                     variant=args.variant,
                                     args=args,
                                     )
            elif self.baseblock == 'mlp':
                gcb = self.BASEBLOCK(in_dim=nfeat,
                                     hidden_dim=nhid,
                                     out_dim=nclass,
                                     n_layers=nbaselayer,
                                     dropout=dropout,
                                     args=args)
            else:  # gcn
                gcb = self.BASEBLOCK(in_features=nfeat,
                                     hidden_features=nhid,
                                     out_dim=nclass,
                                     nbaselayer=nbaselayer,
                                     withbn=withbn,
                                     withloop=withloop,
                                     activation=activation,
                                     dropout=dropout,
                                     dense=False,
                                     args=args,
                                     )
            self.midlayer.append(gcb)
        if baseblock.lower() == 'gcnii':
            # self.ingc = nn.Linear(nfeat, nhid)
            # self.outgc = nn.Linear(nhid, nclass)
            # self.fcs = nn.ModuleList([self.ingc, self.outgc])
            self.params1 = self.midlayer[0].params1
            self.params2 = self.midlayer[0].params2
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, fea, adj, G=None):

        if self.baseblock in ['mlp', 'gcn', 'gcnii']:
            out = self.midlayer[0](input=fea, adj=adj, G=G)
            return out
