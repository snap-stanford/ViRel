import numpy as np
import pickle
import pdb
import ipdb
import torch
import torch.nn as nn
from torch.nn import functional as F

from reasoning_util import get_activation
from loss import intra_inter_loss


def mlp(layers, dim, mlp_act):
    return MLP(layers, dim, dim, dim, mlp_act)

class MaskedMLP(nn.Module):
    def __init__(self, mlp_layers, dim, mlp_act):
        super().__init__()
        
        self.mlp = mlp(mlp_layers, dim, mlp_act)
    
    def forward(self, x, mask):
        mask = mask[:, :, None]
        out = x * mask
        out = self.mlp(out)
        out = out * mask
        return out

### MLP with linear output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, act_name="relu"):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.acts = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.acts.append(get_activation(act_name))
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
    
    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            x_shape_f2 = list(x.shape)[:2]
            h = x.flatten(0, 1)
            for layer in range(self.num_layers - 1):
                h = self.acts[layer](self.batch_norms[layer](self.linears[layer](h)))
            h = self.linears[self.num_layers - 1](h).view(x_shape_f2 + [-1])
            
            return h
