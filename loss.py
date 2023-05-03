import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pdb
import ipdb
import torch_scatter

from util import *

class_loss = nn.CrossEntropyLoss(reduction='none')


def get_data(inpt, name):
    if name == "x":
        keys = ["task_ids", "gt_edge", "edge_mask", "edge_mask_neg"]
    elif name == "out":
        keys = ["logits", "gin_output_sum", "edge_aug", "latent", "latent_aug", "latent_neg", "rel_logprob", "rel_prior_logprob"]
    
    return [inpt[key] if key in inpt else None for key in keys]

def graph_task_loss(args, x, out, step, alphas=None):
    task_ids, gt_edge, edge_mask, edge_mask_neg = get_data(x, "x") 
    logits, gin_output_sum, edge_aug, latent, latent_aug, latent_neg, rel_logprob, rel_prior_logprob = get_data(out, "out") 
    
    block_diag = torch.eq(task_ids[None], task_ids[:, None]).type(torch.float)

    cl_loss, cl_loss_mean = torch.tensor(0), torch.tensor(0)
    intra_diff_mean, inter_diff_mean = torch.tensor(0), torch.tensor(0)
    pairwise_diff = None
    if args.classify:
        cl_loss = class_loss(logits, task_ids)
        cl_loss_mean = cl_loss.mean()
    else:
        margin = (2.0/3) * gin_output_sum.size(-1)
        intra_diff_mean, inter_diff_mean, pairwise_diff = intra_inter_loss2(gin_output_sum, task_ids, margin, step)

    edge_rep_loss = latent_neg_repel_loss(latent, latent_neg, edge_mask, edge_mask_neg, gt_edge)
    # edge_rep_loss = latent_repel_loss(latent, edge_mask, gt_edge)
    
    edge_att_loss = torch.tensor(0) 
    if edge_aug is not None:
        edge_att_loss = latent_attract_loss(latent, latent_aug, edge_mask, gt_edge)  
        edge_rep_loss = (edge_rep_loss + latent_neg_repel_loss(latent_aug, latent_neg, edge_mask, edge_mask_neg, gt_edge))/2.0

    alpha_entr_loss = torch.tensor(0) 
    if alphas is not None:
        alphas = alphas.clip(1e-6, 1.0 - 1e-6)
        alpha_entr_loss = -1 * (alphas * torch.log(alphas) + (1 - alphas) * torch.log(1 - alphas)).mean()

    cl_weight = float(args.cl_weight)
    intra_weight = float(args.intra_weight)
    inter_weight = float(args.inter_weight)
    edge_rep_weight = float(args.edge_rep_weight)
    edge_att_weight = float(args.edge_att_weight)
    ixz_weight = float(args.ixz_weight)
    alpha_entr_weight = float(args.alpha_entr)

    loss = intra_diff_mean * intra_weight + inter_diff_mean * inter_weight + \
           edge_rep_loss * edge_rep_weight + edge_att_loss * edge_att_weight + \
           alpha_entr_loss * alpha_entr_weight + cl_loss_mean * cl_weight

    rel_logprob_mean, rel_prior_logprob_mean = torch.tensor(0), torch.tensor(0)
    if args.is_ixz and rel_logprob is not None:
        rel_logprob_mean, rel_prior_logprob_mean = rel_logprob.mean(), rel_prior_logprob.mean()
        loss = loss + (rel_logprob_mean - rel_prior_logprob_mean) * ixz_weight

    loss_d = dict(
        loss=loss.item(),
        intra_loss=intra_diff_mean.item(),
        inter_loss=inter_diff_mean.item(),
        cl_loss=cl_loss_mean.item(),
        edge_rep_loss=edge_rep_loss.item(),
        edge_att_loss=edge_att_loss.item(),
        rel_logprob=rel_logprob_mean.item(),
        rel_prior_logprob=rel_prior_logprob_mean.item(),
        ixz_bound=(rel_logprob_mean - rel_prior_logprob_mean).item(),
        alpha_entropy_loss=alpha_entr_loss.item(),
    )
    return loss, loss_d, pairwise_diff, block_diag

def intra_inter_loss2(gin_output, task_ids, margin, step):
    task_graph_protos = torch_scatter.scatter(gin_output, task_ids, dim=0, reduce='mean')
    # task_graph_protos_tgt = torch.tensor( 
    #     [[-0.1234, -2.3086, -2.1037],
    #     [-0.5515, -1.3321, -0.6064],
    #     [-1.6965, -0.4579, -0.4828],
    #     [-2.1672, -0.3021, -2.1382],
    #     [-1.1366, -1.4432, -2.2343]], dtype=torch.float32, device=gin_output.device)
      
    rows, cols = torch.triu_indices(task_graph_protos.shape[0], task_graph_protos.shape[0], offset=1)
    task_graph_protos_r = task_graph_protos[rows]
    task_graph_protos_c = task_graph_protos[cols]
    
    inter_diff =  F.relu(margin - (task_graph_protos_r - task_graph_protos_c).abs().sum(-1))
    
    intra_diff = (gin_output - task_graph_protos[task_ids]).abs() # If we do sum(-1) basically a 3x on the weight

    inter_diff_mean = inter_diff.mean()
    intra_diff_mean = intra_diff.mean()

    pairwise_diff = gin_output[:, None] - gin_output[None]
    pairwise_diff = torch.linalg.norm(pairwise_diff, dim=-1)


    # todo: try interleave weight instead of all at once
    if step % 2 == 0: # likely to improve: use momentum targets and still push away intra?
        intra_diff_mean = intra_diff_mean.detach()
    else:
        inter_diff_mean = inter_diff_mean.detach()

    return intra_diff_mean, inter_diff_mean, pairwise_diff

def intra_inter_loss(gin_output, task_ids, margin, step=None):
    # num tasks is max of task_ids + 1
    # gin_output is [18x20]
    
    block_diag = torch.eq(task_ids[None], task_ids[:, None]).float()
    
    # Deprecated ------
    # block_diag_s = get_block_diag(gin_output.size(0), task_ids[-1] + 1, device)
    # sanity = (block_diag - block_diag_s).sum()
    # print(f'sanity {sanity}')
    # if not isinstance(task_ids, int):
    #     B = gin_output.size(0)
    #     block_diag = get_block_diag(B, task_ids, device)
    # -----------

    pos_cnt = block_diag.sum()
    neg_cnt = task_ids.shape[0] ** 2 - pos_cnt

    pairwise_diff = gin_output[:, None] - gin_output[None] #[B=42, B=42, rel_dim=3]
    pairwise_diff = pairwise_diff.abs().sum(dim=-1)
    # pairwise_diff = torch.linalg.norm(pairwise_diff, dim=-1)

    intra_diff = pairwise_diff * block_diag
    inter_diff = F.relu(margin - pairwise_diff) * (1 - block_diag)

    intra_diff_mean = intra_diff.sum()/pos_cnt
    inter_diff_mean = inter_diff.sum()/neg_cnt

    return intra_diff_mean, inter_diff_mean, pairwise_diff

def edge_attract_loss(edges, edges_aug, edge_mask):
    diff = (edges - edges_aug).abs() # L1 or BCE loss?
    
    ea_loss = diff[edge_mask.bool()].mean()  # this or third is fastest
    
    # masked_diff = diff * edge_mask[:, :, None]
    # ea_loss2 = masked_diff.mean(dim=2).sum()/edge_mask.sum()
    
    # masked_diff = diff * edge_mask[:, :, None]
    # ea_loss3 = masked_diff.sum()/(edge_mask.sum() * masked_diff.shape[-1])

    return ea_loss

def latent_attract_loss(latent, latent_aug, edge_mask, gt_edges):
    latent = normalize_embedding(latent[edge_mask.bool()])
    latent_aug = normalize_embedding(latent_aug[edge_mask.bool()])
    
    # Dotprod (not pairwise)
    dotprod = torch.mul(latent, latent_aug).sum(dim=-1)
    ea_loss = 1 - dotprod.mean() #maximize the dotprod
    
    return ea_loss

def latent_repel_loss(latent, edge_mask, gt_edges):
    latent = normalize_embedding(latent[edge_mask.bool()])
    
    rows, cols = torch.triu_indices(latent.shape[0], latent.shape[0], offset=1)
    latent_rows = latent[rows]
    latent_cols = latent[cols]
    
    # gt_edges = gt_edges[edge_mask.bool()]
    # latent0 = latent[gt_edges == 0]
    # latent1 = latent[gt_edges == 1]
    # latent2 = latent[gt_edges == 2]
    # latent3 = latent[gt_edges == 3]
    
    pw_dotprod = (latent_rows * latent_cols).sum(dim=-1)
    er_loss = pw_dotprod.mean() #minimize
    del pw_dotprod

    return er_loss

def latent_neg_repel_loss(latent, latent_neg, edge_mask, edge_mask_neg, gt_edges):
    er_loss = 0
    latent = normalize_embedding(latent[edge_mask.bool()])
    latent_neg = normalize_embedding(latent_neg[edge_mask_neg.bool()]) #try not .detach() this?

    # pw_dotprod_old = (latent[:, None] * latent_neg[None]).sum(dim=-1)
    pw_dotprod = torch.mm(latent, latent_neg.T)
    er_loss = pw_dotprod.mean() #minimize

    return er_loss


def edge_repel_loss(edges, edge_mask):
    # either can do EVERY pairwise edge repulsion, or a random 2-pair repulsion
    
    # mask for non-edges (2 obj case)
    edges = edges[edge_mask.bool()]
    # edges = edges.flatten(0, 1) # [252, 3, 3] -> [736 x 3]
    
    rows, cols = torch.triu_indices(edges.shape[0], edges.shape[0], offset=1)
    edge_rows = edges[rows]
    edge_cols = edges[cols]
    # bce = ((1 - edge_rows) - edge_cols.detach()).abs() #.pow(2)
    er = 1 - (edge_rows - edge_cols).abs()
    # bce = F.binary_cross_entropy(1 - edge_rows, edge_cols.detach(), reduction="none") # could also try L2 but seems messy as well
    er_val, _ = er.min(dim=-1)
    er_loss = er_val.mean() 
    
    return er_loss

def edge_neg_repel_loss(edges, edges_neg, edge_mask, edge_mask_neg):
    edges = edges[edge_mask.bool()]
    edges_neg = edges_neg.detach()[edge_mask_neg.bool()]
    
    #debugging
    
    # edgesv = edges.round()[:30]
    # edgesnv = edges_neg.round()[:30]
    
    pairwise_diff = 1 - (edges[:, None] - edges_neg[None]).abs()
    
    # pwd_loss = pairwise_diff.mean()
    pwd_val, pwd_locs = pairwise_diff.min(dim=-1)
    pwd_loss = pwd_val.mean() 
    
    return pwd_loss

def normalize_embedding(embeddings, eps=1e-12):
  """Normalizes embedding by L2 norm.
  This function is used to normalize embedding so that the
  embedding features lie on a unit hypersphere.
  Args:
    embeddings: An N-D float tensor with feature embedding in
      the last dimension.
  Returns:
    An N-D float tensor with the same shape as input embedding
    with feature embedding normalized by L2 norm in the last
    dimension.
  """
  norm = torch.norm(embeddings, dim=-1, keepdim=True)
  norm = torch.where(torch.ge(norm, eps),
                     norm,
                     torch.ones_like(norm).mul_(eps))
  return embeddings / norm