import numpy as np
import pickle
import time
import pdb
import ipdb
import torch
import torch.nn as nn
from torch.nn import functional as F

from reasoning_util import get_activation

from models.cnn import ObjCatRelEnc, CatObjRelEnc
from models.gin import GIN_Task

def update_default_model(args):
    args.encoder_act = "leakyrelu"
    args.gin_act = "leakyrelu0.2"
    args.mlp_act = "leakyrelu"
    args.mlp_act_gnn = "leakyrelu0.2"
    args.in_channels = 9
    args.rel_dim_m = args.rel_dim + int(args.softmax_rel)
    args.gin_mlp_layers = 3
    args.gin_layers = 2 # TODO tune this value (did some tuning already)

def get_gnn_model(args):
    update_default_model(args)
    return GNNTaskModelAug(args)

class GNNTaskModel(nn.Module):
    def __init__(self, ag):
        super().__init__()

        if ag.mlp_concat:
            self.obj2rel_enc = ObjCatRelEnc(ag)
        else:
            self.obj2rel_enc = CatObjRelEnc(ag)
        
        self.gin_task = GIN_Task(ag)

        if ag.freeze_gin:
            for name, param in self.gin_task.named_parameters():
                if name != "ex2alpha.weight":
                    param.requires_grad = False
        
        self.is_classify = ag.classify
        if ag.classify:
            self.classify_act = get_activation(ag.gin_act) if ag.gin_last_act else nn.Identity()
            self.classify = nn.Linear(ag.rel_dim_m, ag.total_tasks) 

    def forward(self, x, x_key, softmax, only_edge):
        inpt, mask, task_ids, ex_ids, shuf_rel_idx = self.get_x_data(x, x_key)
        rel_matrix_raw, latent_matrix, obj_encoded, rel_logprob, rel_prior_logprob = self.obj2rel_enc(inpt)
        
        rel_matrix = torch.softmax(rel_matrix_raw, dim=-1) if softmax else rel_matrix_raw # torch.sigmoid(rel_matrix)
            
        out = {
            "edge" + x_key: rel_matrix,
            "latent" + x_key: latent_matrix,
            "obj_enc" + x_key: obj_encoded,
            "rel_logprob" + x_key: rel_logprob,
            "rel_prior_logprob" + x_key: rel_prior_logprob
        }
        # rel_matrix = normalized_normal_pdf(rel_matrix_raw)
        if only_edge:
            return out
    
        # Compute GIN
        gin_output_sum, gin_output = self.gin_task(rel_matrix, mask, task_ids, ex_ids, shuf_rel_idx)

        # Logits on GIN
        if self.is_classify:
            logits = self.classify(self.classify_act(gin_output_sum))
            out['logits' + x_key] = logits
            out['task_id_pred' + x_key] = logits.argmax(dim=1)
        
        out['gin_output' + x_key] = gin_output
        out['gin_output_sum' + x_key] = gin_output_sum

        return out

    def get_x_data(self, x, x_key):
        keys = ["mask_img" + x_key, 'edge_mask', 'task_ids', 'ex_ids', 'shuf_rel_idx']
        return [x[key] for key in keys]


class GNNTaskModelAug(GNNTaskModel):
    def forward(self, x, softmax, only_edge, aug, neg):
        
        out = super().forward(x, '', softmax, only_edge)
        
        if aug:
            assert x['mask_img_augs'] is not None
            assert x['mask_img_augs'].shape[1] == 1, "More than 1 augmentation not currently supported"
            x['mask_img_aug'] = x['mask_img_augs'][:, 0]

            out_aug = super().forward(x, '_aug', softmax, only_edge=True)
            out.update(out_aug)
        
        if neg:
            out_neg = super().forward(x, '_neg', softmax, only_edge=True)
            out.update(out_neg)

        return out
 