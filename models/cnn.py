import numpy as np
import pickle
import pdb
import ipdb
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal

from reasoning_util import get_activation
from pytorch_net.net import Mixture_Gaussian_reparam

from util import kaiming_init

from models.mlp import MaskedMLP, MLP, mlp

class ObjCatRelEnc(nn.Module):
    # Define a CNN for 1 x w x h
    # Conv2d args: in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
    def __init__(self, ag):
        super().__init__()
        
        # B = num tasks * num examples per task
        # D = B * num_objs, C = num_colors = 9
        # D, C, 8, 8
        def get_unsup_encoder_8x8():
            fst_dim = 16
            sec_dim = 36
            thd_dim = 144
            frt_dim = 576
            # fst_dim = 30
            # sec_dim = 70
            # thd_dim = 300
            # frt_dim = 1100
            fth_dim = 500
            sth_dim = 250

            return nn.Sequential(
                nn.Conv2d(ag.in_channels, fst_dim, 3, 1),    # D, 16, 6, 6
                nn.BatchNorm2d(fst_dim),
                get_activation(ag.encoder_act),
                nn.Conv2d(fst_dim, sec_dim, 3, 1),        # D, 36, 4, 4
                nn.BatchNorm2d(sec_dim),
                get_activation(ag.encoder_act),
                nn.Conv2d(sec_dim, thd_dim, 3, 1),          # D, 144,  2, 2
                nn.BatchNorm2d(thd_dim),
                get_activation(ag.encoder_act),
                nn.Conv2d(thd_dim, frt_dim, 2, 1),          # D, 576, 1, 1
                nn.BatchNorm2d(frt_dim),
                get_activation(ag.encoder_act),
                nn.Flatten(),                                # D, 576
                nn.Linear(frt_dim, fth_dim),                  # D, 500
                get_activation(ag.encoder_act),
                nn.Linear(fth_dim, sth_dim),                  # D, 250
                get_activation(ag.encoder_act),
                nn.Linear(sth_dim, ag.obj_dim)              # D, obj_dim=100
            )

        def get_unsup_encoder_16x16():
            fst_dim = 25
            sec_dim = 50
            thd_dim = 144
            frt_dim = 576
            fth_dim = 500
            sth_dim = 250

            return nn.Sequential(
                nn.Conv2d(ag.in_channels, fst_dim, 3, 2),    # D, 16, 7, 7
                nn.BatchNorm2d(fst_dim),
                get_activation(ag.encoder_act),
                nn.Conv2d(fst_dim, sec_dim, 3, 1),        # D, 36, 5, 5
                nn.BatchNorm2d(sec_dim),
                get_activation(ag.encoder_act),
                nn.Conv2d(sec_dim, thd_dim, 3, 1),          # D, 144,  3, 3
                nn.BatchNorm2d(thd_dim),
                get_activation(ag.encoder_act),
                nn.Conv2d(thd_dim, frt_dim, 3, 1),          # D, 576, 1, 1
                
                nn.BatchNorm2d(frt_dim),
                get_activation(ag.encoder_act),
                nn.Flatten(),                                # D, 576
                nn.Linear(frt_dim, fth_dim),                  # D, 500
                get_activation(ag.encoder_act),
                nn.Linear(fth_dim, sth_dim),                  # D, 250
                get_activation(ag.encoder_act),
                nn.Linear(sth_dim, ag.obj_dim)              # D, obj_dim=100
            )
        
        def get_sup_encoder():
            zth_dim = 16
            fst_dim = 18
            sec_dim = 36
            thd_dim = 144
            frt_dim = 576
            # fst_dim = 30
            # sec_dim = 70
            # thd_dim = 300
            # frt_dim = 1100
            fth_dim = 500

            return nn.Sequential(
                nn.Conv2d(ag.in_channels, zth_dim, 3, 1, 1),    # D, 16, 8, 8
                nn.BatchNorm2d(zth_dim),
                get_activation(ag.encoder_act),
                
                nn.Conv2d(zth_dim, fst_dim, 3, 1),    # D, 16, 6, 6
                nn.BatchNorm2d(fst_dim),
                get_activation(ag.encoder_act),
                
                nn.Conv2d(fst_dim, fst_dim, 3, 1, 1),    # D, 16, 6, 6
                nn.BatchNorm2d(fst_dim),
                get_activation(ag.encoder_act),
                
                nn.Conv2d(fst_dim, sec_dim, 3, 1),        # D, 36, 4, 4
                nn.BatchNorm2d(sec_dim),
                get_activation(ag.encoder_act),
                
                nn.Conv2d(sec_dim, sec_dim, 3, 1, 1),        # D, 36, 4, 4
                nn.BatchNorm2d(sec_dim),
                get_activation(ag.encoder_act),
                
                nn.Conv2d(sec_dim, thd_dim, 3, 1),          # D, 144,  2, 2
                nn.BatchNorm2d(thd_dim),
                get_activation(ag.encoder_act),
                                
                nn.Conv2d(thd_dim, frt_dim, 2, 1),          # D, 576, 1, 1
                nn.BatchNorm2d(frt_dim),
                get_activation(ag.encoder_act),
                nn.Flatten(),                                # D, 576
                nn.Linear(frt_dim, fth_dim),                  # D, 500
                get_activation(ag.encoder_act),
                nn.Linear(fth_dim, ag.obj_dim)              # D, obj_dim=100
            )
        
        self.object_enc = get_unsup_encoder_8x8() if ag.canvas_size == 8 else get_unsup_encoder_16x16()
        self.object_enc.apply(kaiming_init)

        self.triu_indices = torch.triu_indices(ag.padding_objs, ag.padding_objs, offset=1)
        
        self.is_ixz = ag.is_ixz
        self.ixz_gin_mean = ag.ixz_gin_mean
        self.reparam_mode = ag.reparam_mode
        self.dist_prior = Mixture_Gaussian_reparam(
            Z_size=ag.rel_dim_m,
            n_components=ag.num_rels,
            mean_scale=0.1 if ag.reparam_mode == "diag" else 2.0,
            scale_scale=0.1,
            # Mode:
            is_reparam=False,
            reparam_mode=ag.reparam_mode,
        ) if ag.is_ixz else None

        self.rel_dim = ag.rel_dim_m
        #         self.obj2rel = nn.Linear(obj_dim * 2, ag.rel_dim_m)
        self.obj2rel_layers = 3
        self.multi_head_rel = ag.multi_head_rel
    
        if self.multi_head_rel:
            self.obj2rels = nn.ModuleList()
            for _ in range(ag.rel_dim_m):
                self.obj2rels.append(MLP(self.obj2rel_layers, ag.obj_dim * 2, ag.obj_dim * 2, 1, ag.mlp_act))
        else:
            if ag.is_ixz:
                if ag.reparam_mode == "diag":
                    self.obj2rel = MLP(self.obj2rel_layers + 1, ag.obj_dim * 2, ag.obj_dim * 2, ag.rel_dim_m * 2, ag.mlp_act)
                elif ag.reparam_mode == "diagg":
                    self.obj2rel = MLP(self.obj2rel_layers + 1, ag.obj_dim * 2, ag.obj_dim * 2, ag.rel_dim_m, ag.mlp_act)
                else:
                    raise
            else:
                self.obj2latent = MLP(self.obj2rel_layers, ag.obj_dim * 2, ag.obj_dim * 2, ag.obj_dim * 2, ag.mlp_act) #Try doing ER/EA after this
                self.latent2relweight = torch.nn.Parameter(torch.FloatTensor(ag.rel_dim_m, ag.obj_dim * 2))
                nn.init.xavier_uniform_(self.latent2relweight)
                
                # self.latent2rel = F.linear(latent2relweight)
                # nn.utils.weight_norm()
                # self.latent2rel = nn.Linear(ag.obj_dim * 2, ag.rel_dim_m, bias=False)

                # self.obj2latent = MLP(self.obj2rel_layers, ag.obj_dim * 2, ag.obj_dim * 2, ag.rel_dim_m, mlp_act) #Try doing ER/EA after this
    
    def forward(self, x):
        shape = list(x.shape)

        # Encode each mask into latent dimension
        enc = self.object_enc(x.flatten(0, 1))
        obj_encoded = enc.view([shape[0], shape[1], -1]) #[B, num_objs=3, outdim=50])

        rel_matrix, latent_matrix, rel_logprob, rel_prior_logprob = self.edge_feature_mlp(obj_encoded)
          
        # It's up to the downstream to mask the rel_matrix
        return rel_matrix, latent_matrix, obj_encoded, rel_logprob, rel_prior_logprob
    
    def edge_feature_mlp(self, node_embeds):
        rows, cols = self.triu_indices

        pairwise_cat = torch.cat([node_embeds[:, rows], node_embeds[:, cols]], dim=-1)
        
        out = self.obj2relMLP(pairwise_cat)
        return out
    
    def obj2relMLP(self, pairwise_cat):
        if self.multi_head_rel:
            outs = [mlp(pairwise_cat) for mlp in self.obj2rels]
            return torch.cat(outs, dim=-1)
        else:
            rel_logprob = None
            rel_prior_logprob = None
            if self.is_ixz:
                # TODO: rename obj2rel to obj2latent
                latent = self.obj2rel(pairwise_cat)  # pairwise_cat: [504, 3, 200], latent: [504, 3, 40]
                if self.reparam_mode == "diag":
                    loc, logscale = latent[..., :self.rel_dim], latent[..., self.rel_dim:]
                    scale = torch.exp(logscale)
                elif self.reparam_mode == "diagg":
                    loc = latent
                    scale = torch.ones_like(loc)
                else:
                    raise
                dist = Normal(loc, scale)
                rel = dist.rsample() if self.training and not self.ixz_gin_mean else loc
                if self.training:
                    rel_sample = dist.rsample() if self.ixz_gin_mean else rel
                    rel_logprob = dist.log_prob(rel_sample)
                    rel_prior_logprob = self.dist_prior.log_prob(rel_sample)
            else:
                latent = self.obj2latent(pairwise_cat)  # pairwise_cat: [504, 3, 200]
                rel = F.linear(F.normalize(latent, dim=-1), F.normalize(self.latent2relweight, dim=-1)) 
                #self.latent2rel(latent)
            return rel, latent, rel_logprob, rel_prior_logprob

    def edge_feature_diff(self, node_embeds):
        rows, cols = self.triu_indices

        edge_feature_matrix = node_embeds[:, :, None] - node_embeds[:, None]
        pairwise_diff = edge_feature_matrix[:, rows, cols]
        
        # pairwise_diff = node_embeds[:, rows] - node_embeds[:, cols]
        # second is eq. the same but 4x slower (0.000429 vs 0.0013)
        return pairwise_diff

class CatObjRelEnc(nn.Module):
    def __init__(self, in_channels, obj_dim, rel_dim, act_name):            
        super().__init__()
        # D = B * num_objs, C = 2 * num_colors = 18 (because concat at channel dimension)
        # D, C, 8, 8
        
        # first_dim = 16
        # sec_dim = 36
        # thd_dim = 144
        # frt_dim = 576
        
        # Params during unsup tuning
        fst_dim = 32
        sec_dim = 72
        thd_dim = 300 #300
        frt_dim = 600 #150
        fth_dim = 600 #500
        
        self.object2rel_enc = nn.Sequential(
            nn.Conv2d(2 * in_channels, fst_dim, 3, 1),    # D, 32, 6, 6
            nn.BatchNorm2d(fst_dim),
            get_activation(act_name),
            nn.Conv2d(fst_dim, sec_dim, 3, 1),        # D, 72, 4, 4
            nn.BatchNorm2d(sec_dim),
            get_activation(act_name),
            nn.Conv2d(sec_dim, thd_dim, 3, 1),          # D, 300,  2, 2
            nn.BatchNorm2d(thd_dim),
            get_activation(act_name),
            nn.Conv2d(thd_dim, frt_dim, 2, 1),          # D, 1150, 1, 1
            nn.BatchNorm2d(frt_dim),
            get_activation(act_name),
            nn.Flatten(),                                # D, 1150
            nn.Linear(frt_dim, fth_dim),                  # D, 500
            get_activation(act_name),
            nn.Linear(fth_dim, 2 * obj_dim),              # D, 2 * obj_dim=200
            # get_activation(act_name),
            # nn.Linear(2 * obj_dim, 2 * obj_dim),
            # get_activation(act_name),
            # nn.Linear(2 * obj_dim, 2 * obj_dim),
            get_activation(act_name),
            nn.Linear(2 * obj_dim, rel_dim)              # D, rel_dim=3 or 4
        )
    
        # Params worked for Supervised
        # fst_dim = 32
        # sec_dim = 72
        # thd_dim = 300
        # frt_dim = 1150
        # fth_dim = 500
        
        # Encoder worked for Supervised
        # self.object2rel_enc = nn.Sequential(
        #     nn.Conv2d(2 * in_channels, fst_dim, 3, 1),    # D, 32, 6, 6
        #     nn.BatchNorm2d(fst_dim),
        #     get_activation(act_name),
        #     nn.Conv2d(fst_dim, sec_dim, 3, 1),        # D, 72, 4, 4
        #     nn.BatchNorm2d(sec_dim),
        #     get_activation(act_name),
        #     nn.Conv2d(sec_dim, thd_dim, 3, 1),          # D, 300,  2, 2
        #     nn.BatchNorm2d(thd_dim),
        #     get_activation(act_name),
        #     nn.Conv2d(thd_dim, frt_dim, 2, 1),          # D, 1150, 1, 1
        #     nn.BatchNorm2d(frt_dim),
        #     get_activation(act_name),
        #     nn.Flatten(),                                # D, 1150
        #     nn.Linear(frt_dim, fth_dim),                  # D, 500
        #     get_activation(act_name),
        #     nn.Linear(fth_dim, 2 * obj_dim),              # D, 2* obj_dim=200
        #     get_activation(act_name),
        #     nn.Linear(2 * obj_dim, rel_dim)              # D, rel_dim=3 or 4
        # )
        
        self.object2rel_enc.apply(kaiming_init)
        
        
    def forward(self, x):
        rows, cols = self.triu_indices
        obj_cat = torch.cat([x[:, rows], x[:, cols]], dim=2)
        
        shape = list(obj_cat.shape)
        
        edge_feature_matrix = self.object2rel_enc(obj_cat.flatten(0, 1))
        edge_feature_matrix = edge_feature_matrix.view([shape[0], shape[1], -1])
        
        return edge_feature_matrix, None