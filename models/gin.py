from inspect import isclass
import numpy as np
import pickle
import time
import pdb
import ipdb
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GCNConv, DenseGINConv

from reasoning_util import get_activation

from util import get_adj_t_matrix, get_num_edges
from loss import intra_inter_loss

from models.mlp import MaskedMLP, MLP, mlp

class GIN_Task(nn.Module):
    def __init__(self, ag):
        super().__init__()

        self.is_inner_loop = ag.is_inner_loop
        self.is_lookup_mask = ag.is_lookup_mask
        
        if ag.is_lookup_mask:
            self.ex2alpha = nn.Embedding(ag.total_tr_examples, get_num_edges(ag.padding_objs), dtype=torch.float32)
            init_val = 1.0
            nn.init.constant_(self.ex2alpha.weight, init_val)
        elif ag.is_inner_loop:
            self.num_inner_loop = ag.num_inner_loop # for "MAML"
            self.register_buffer('inner_lr', torch.tensor(ag.inner_lr, dtype=torch.float32))
            self.register_buffer('save_il_print', torch.tensor(0, dtype=torch.float32))
            # self.alphas = torch.nn.Parameter(torch.ones(get_num_edges(PADDING_OBJS), dtype=torch.float32))
            # self.register_buffer('alphas', torch.ones(get_num_edges(PADDING_OBJS), dtype=torch.float32))
            # self.save_il_print = 0.2
        
        self.register_buffer('adj_t', torch.tensor(get_adj_t_matrix(ag.padding_objs), dtype=torch.float32))
        
        if ag.gin_layers > 0:
            self.graph_enc = GIN_Enc(ag.gin_layers, ag.gin_mlp_layers, ag.rel_dim_m, self.adj_t, ag.gin_act, ag.mlp_act_gnn)
        else:
            self.graph_enc = MaskedMLP(ag.gin_mlp_layers, ag.rel_dim_m, ag.mlp_act_gnn)   

        # Todo: init GNN as well?

    def forward(self, rel_matrix_s, mask, task_ids=None, ex_ids=None, shuf_rel_idx=None):
        # assert not (self.is_inner_loop and task_ids is None)
        if self.is_lookup_mask and self.training:
            return self.forward_lkup(rel_matrix_s, mask, ex_ids, shuf_rel_idx)
        elif self.is_inner_loop:
            return self.forward_il(rel_matrix_s, mask, task_ids)
        return self.forward_(rel_matrix_s, mask)

    def forward_(self, rel_matrix_s, mask):
        gin_output = self.graph_enc(rel_matrix_s, mask)
        gin_output_sum = gin_output.sum(dim=1)

        return gin_output_sum, gin_output

    def forward_lkup(self, rel_matrix_s, mask, ex_ids, shuf_rel_idx):
        alphas = self.ex2alpha(ex_ids)
        alphas = torch.gather(alphas, 1, shuf_rel_idx)

        gin_output = self.graph_enc(rel_matrix_s, mask, alphas)
        gin_output_sum = gin_output.sum(dim=1)

        return gin_output_sum, gin_output

    def forward_il(self, rel_matrix_s, mask, task_ids):
        # Do inner loop here
        
        # with torch.enable_grad():
        # self.alphas.data = self.alphas.clip(min=0.0, max=1.0)
        il_alpha = torch.ones_like(mask, dtype=torch.float32, requires_grad=True)
        il_scalar = 1 #0.5
        il_alpha = il_alpha * il_scalar
        
        for _ in range(self.num_inner_loop):
            gin_output = self.graph_enc(rel_matrix_s, mask, il_alpha)
            gin_output_sum = gin_output.sum(dim=1)
            
            intra_loss, inter_loss, _ = intra_inter_loss(gin_output_sum, task_ids, (2.0/3) * gin_output_sum.size(-1)) #Margin only needed for inter
            grad_loss = intra_loss # + inter_loss

            alpha_grad = torch.autograd.grad(grad_loss, il_alpha, create_graph=self.training)[0]
            il_alpha = il_alpha - self.inner_lr * alpha_grad       
            il_alpha = il_alpha.clip(min=0.0, max=1.0)

        self.save_il_print = il_alpha
        gin_output = self.graph_enc(rel_matrix_s, mask, il_alpha)
        gin_output_sum = gin_output.sum(dim=1)

        return gin_output_sum, gin_output

class GIN_Enc(nn.Module):
    def __init__(self, gin_layers, mlp_layers, dim, adj_t, act_name, mlp_act):
        super().__init__()
        
        self.register_buffer('adj_t', adj_t)
        self.gins = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.bns = nn.ModuleList()

        if False and gin_layers == 2:
            # HACK only testing 3->20
            self.gins.append(DenseGINConvCat(MLP(mlp_layers, 2 * dim, 20, 20, mlp_act)))
            self.gins.append(DenseGINConvCat(MLP(mlp_layers, 40, 40, 20, mlp_act)))
            self.bns.append(nn.BatchNorm1d(20))
            self.acts.append(get_activation(act_name))
        else:
            for i in range(gin_layers - 1):
                self.gins.append(DenseGINConvCat(MLP(mlp_layers, 2 * dim, 2 * dim, dim, mlp_act)))
                # self.gins.append(DenseGINConv1x1Conv(nn.Conv2d(2, 1, 1, 1))) # 1x1 conv from 2 -> 1
                self.bns.append(nn.BatchNorm1d(dim))
                self.acts.append(get_activation(act_name))
            self.gins.append(DenseGINConvCat(MLP(mlp_layers, 2 * dim, 2 * dim, dim, mlp_act)))
            # self.gins.append(DenseGINConv1x1Conv(nn.Conv2d(2, 1, 1, 1)))
            
    def forward(self, x, mask, alphas=None):
        out = x
        mask = mask[:, :, None]
        alpha_mask = mask * alphas[:, :, None] if alphas is not None else mask

        # for now: gate every layer by alpha, also helps boost gradient

        # Gate the x output by alphas and masks
        out = out * alpha_mask
        
        for gin, bn, act in zip(self.gins, self.bns, self.acts):       
            out = gin(out, self.adj_t, add_loop=False)
            out_shape = out.shape
            out = out.flatten(0, 1)
            # out = bn(out) # Todo: Group norm may be better
            # Hmm. maybe not a good idea because each unit should not be affecting each other.
            out = act(out)
            out = out.view(out_shape)
            # out = out * alpha_mask

        out = self.gins[-1](out, self.adj_t, add_loop=False)
        # out = out * alpha_mask # Only gate by mask for last output
        
        return out
            
class DenseGINConvCat(DenseGINConv):
    def forward(self, x, adj, mask=None, add_loop=True):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        out = torch.matmul(adj, x)
        if add_loop:
            out = (1 + self.eps) * x + out
        
        out = torch.cat([x, out], dim=-1)
        out = self.nn(out)

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

class DenseGINConv1x1Conv(DenseGINConv):
    def forward(self, x, adj, mask=None, add_loop=True):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        out = torch.matmul(adj, x)
        if add_loop:
            out = (1 + self.eps) * x + out

        out = torch.cat([x[:, None], out[:, None]], dim=1)
        out = self.nn(out).squeeze(1) #1x1 conv

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out    
