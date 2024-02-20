import torch
import torch.nn as nn
import torch.nn.functional as F


###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim,out_dim, n_layers,
                 dropout=0.5,
                 activation=F.relu,
                 withbn=True, withloop=True,
                 aggrmethod="nores", dense=False, res=False, incidence_v=None, incidence_e=None,
                 args=None):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()
        self.in_features = in_dim
        self.hiddendim = hidden_dim
        self.nhiddenlayer = n_layers
        self.sigma = activation
        self.dropout=nn.Dropout(p=dropout)
        self.hiddenlayers = nn.ModuleList()
        self.out_dim=out_dim
        self.args = args



        self.num_layers = n_layers

        if n_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif n_layers == 1:
            self.linear_or_not = True  # default is linear model
            #Linear model
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(in_dim, hidden_dim))
            for layer in range(n_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, out_dim))

            for layer in range(n_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, input, adj=None,G=None):
        x=input
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):

                h = self.sigma(self.batch_norms[layer](self.linears[layer](h)))
                # self.dropout(h)
            return self.linears[self.num_layers - 1](h)

    def get_outdim(self):
        return self.out_dim