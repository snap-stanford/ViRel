import torch
from torch import nn
from torch.nn import functional as F
import random
import os
import pdb
import numpy as np
from functools import lru_cache
from tensorboardX import SummaryWriter
import itertools
import pickle as pkl
from tsne_torch import TorchTSNE as TSNE
import glob
import shutil
from easydict import EasyDict as edict
import argparse
import ipdb
import yaml
import networkx as nx
from numbers import Number

from args import write_yaml

from pytorch_net.util import record_data, draw_nx_graph

def load(log_dir, checkpoint_step, model, optimizer, scheduler):
    """Loads a checkpoint.
    Args:
        checkpoint_step (int): iteration of checkpoint to load
    Raises:
        ValueError: if checkpoint for checkpoint_step is not found
    """
    target_path = (
        f'{os.path.join(log_dir, "state")}'
        f'{checkpoint_step}.pt'
    )
    if os.path.isfile(target_path):
        state = torch.load(target_path)
        model.load_state_dict(state['network_state'], strict=False)
        optimizer.load_state_dict(state['optimizer_state'])
        if is_scheduler(scheduler):
            if 'scheduler_state' in state:
                scheduler.load_state_dict(state['scheduler_state'])
            else:
                print("'scheduler_state' is not in state.")
        print(f"Loaded checkpoint iteration {checkpoint_step}, epoch {state['epoch']}.")
        return state
    
    raise ValueError(
        f'No checkpoint for iteration {checkpoint_step} found.'
        )

def save(log_dir, checkpoint_step, model, optimizer, scheduler, store_last, aux={}):
    """Saves network and optimizer state_dicts as a checkpoint.
    Args:
        checkpoint_step (int): iteration to label checkpoint with
    """
    save_dict = dict(network_state=model.state_dict(),
                     optimizer_state=optimizer.state_dict())
    if is_scheduler(scheduler):
        save_dict["scheduler_state"] = scheduler.state_dict()
    save_dict.update(aux)
    torch.save(
        save_dict,
        f'{os.path.join(log_dir, "state")}{checkpoint_step}.pt'
    )
    
    if store_last:
        with open(f'{os.path.join(log_dir, "chkpt_step")}.p', 'wb') as f:
            pkl.dump(checkpoint_step, f)
    print(f'Saved checkpoint, step: {checkpoint_step}.')

def load_checkpoint_step(log_dir):
    fname = f'{os.path.join(log_dir, "chkpt_step")}.p'
    return load_pkl(fname, -1)
    # return obj if obj is not None else -1

def load_pkl(fname, default=None):
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            return pkl.load(f)
    return default


def set_seed(random_seed):
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    
def get_edge_configs(numRels, softmax):
    edges_bag = []
    if softmax:
        edges_bag = torch.eye(numRels + 1).int().tolist()
    
    io =  (0, 1)
    edges_bag = itertools.product(io, repeat=numRels)
    
    return tuple((itertools.permutations(edges_bag, numRels + 1)))
    
@lru_cache(maxsize=None)
def get_block_diag(B, num_tasks, device):
    examples_per_task = B // num_tasks
    assert examples_per_task > 0
    block_diag = torch.eye(num_tasks)[:, :, None, None] * torch.ones(examples_per_task, examples_per_task)
    block_diag = block_diag.permute(0, 2, 1, 3).reshape(B, B).to(device).detach()
    return block_diag    

# Data utils
def get_num_edges(num_objs): #num_objs=PADDING_OBJS
    return (num_objs * (num_objs - 1))//2            
                
def get_adj_t_matrix(length): #=PADDING_OBJS
    rows, cols = torch.triu_indices(length, length, offset=1)
    rows, cols = rows.tolist(), cols.tolist()
    adj_t = []
    for r, c in zip(rows, cols):
        next_line = []
        first = set([r, c])
        for r2, c2 in zip(rows, cols):
            ret = 0
            if r2 == r and c2 == c:
                pass
            elif r2 in first or c2 in first:
                ret = 1
            next_line.append(ret)
        adj_t.append(next_line)
    return adj_t

@lru_cache(maxsize=None)
def get_valid_edges(num_pads, length): #=PADDING_OBJS
    rows, cols = torch.triu_indices(length, length, offset=1)
    valid_nodes = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        valid_nodes.append(int(r < length - num_pads and c < length - num_pads))
    return valid_nodes

class DummyScheduler():
    def step(self):
        pass

def is_scheduler(scheduler):
    return not isinstance(scheduler, DummyScheduler)

def get_scheduler(conf, optimizer, T_max):
    if conf == "cos":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    return DummyScheduler()

def detach_to(x, device):
    x_ = {}
    for k, v in x.items():
        x_[k] = None if v is None else v.detach().to(device)
    return x_

# From yilun du's
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normalized_normal_pdf(x, sigma=1.0):
    out = x / sigma
    out = torch.pow(out, 2)
    out = -1.0 * out / 2.0
    out = torch.exp(out)
    return out

def model_parallel(model, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        if args.ddp:
            if args.local_rank >= 0:
                torch.cuda.set_device(args.local_rank) 
                device = torch.device(f"cuda:{args.local_rank}")
            torch.distributed.init_process_group(backend='nccl', rank=args.local_rank, world_size=torch.cuda.device_count())
            torch.set_num_threads(os.cpu_count()//(torch.cuda.device_count() * 2))
            model.to(device)
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            model =  torch.nn.parallel.DataParallel(model)
            model.to(device)
    return model, device

def get_nx_graph(graph, is_directed=True, isplot=False):
    g = nx.DiGraph() if is_directed else nx.Graph()
    for item in graph:
        if isinstance(item[0], Number) or isinstance(item[0], str):
            g.add_node(item[0], type=item[1], E=item[2] if len(item) > 2 else None)
        elif isinstance(item[0], tuple):
            src, dst = item[0]
            g.add_edge(
                src,
                dst,
                type=item[1],
                E=item[2] if len(item) > 2 else None,
            )
    if isplot:
        draw_nx_graph(g)
    return g

def backup_py(py_dir):
    src_dir = "/dfs/scratch1/dzeng/reasoning/experiments/relation_analogy_py/*.py"
    py_files = glob.glob(os.path.join(src_dir))

    for file in py_files:
        shutil.copy2(file, py_dir)

def log_record(loss_d, type, epoch, step, writer, data_record):
    # writer.add_scalar('train/loss', loss.item(), i)
    # writer.add_scalar('train/intra_loss', intra_m, i)
    # writer.add_scalar('train/inter_loss', inter_m, i)
    # writer.add_scalar('train/cl_loss', cl_m, i)
    # writer.add_scalar('train/edge_rep_loss', edge_rep, i)
    # writer.add_scalar('train/edge_att_loss', edge_att, i)
    # writer.add_scalar('train/rel_logprob', rlp_mean, i)
    # writer.add_scalar('train/rel_prior_logprob', rplp_mean, i)
    # writer.add_scalar('train/ixz_bound', rlp_mean - rplp_mean, i)
    # writer.add_scalar('train/alpha_entropy_loss', alpha_entr, i)

    # record_data(data_record, [epoch, i, "train", loss.item(), intra_m, inter_m, edge_rep, edge_att, rlp_mean, rplp_mean, rlp_mean-rplp_mean],
    #     ["epoch", "step", "type", "loss", "inter_pair_loss", "intra_pair_loss", "edge_rep_loss", "edge_att_loss", "rel_logprob", "rel_prior_logprob", "ixz_bound"])

    rd_list = [epoch, step, type]
    rd_desc = ["epoch", "step", "type"]
    for k, v in loss_d.items():
        writer.add_scalar(f'{type}/{k}', v, step)
        rd_list.append(v)
        rd_desc.append(k)

    record_data(data_record, rd_list, rd_desc)

def log_writer(args):
    base = args.base_log
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'lr-{args.learning_rate}.batch_size-{args.task_batch_size}.epoch-{args.num_epochs}.seed-{args.random_seed}.tr_aug-{args.train_aug}.num_augs-{args.num_augs}.tr_mask_shuf-{args.train_mask_shuffle}.tr_data_shuf-{args.train_data_shuffle}.mh_rel-{args.multi_head_rel}.supervised-{args.supervised}.mlp_concat-{args.mlp_concat}.sftm_rel-{args.softmax_rel}.rel_dim-{args.rel_dim}.is_ixz-{args.is_ixz}.ixz_gin_mean-{args.ixz_gin_mean}.ixz_reparam-{args.reparam_mode}.intra-{args.intra_weight}.inter-{args.inter_weight}.er-{args.edge_rep_weight}.ea-{args.edge_att_weight}.ixz-{args.ixz_weight}/run-{args.run_id}'    
    log_dir = os.path.join(base, log_dir)
    py_dir = os.path.join(log_dir, "py")
    
    print(f'log_dir: {log_dir}')
    path_existed = os.path.exists(log_dir)
    if path_existed and not args.force_overwrite:
        raise ValueError(f"Log dir already exists at: {log_dir}")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(py_dir, exist_ok=True)
    
    if not path_existed or args.force_py_backup:
        # Copy python files
        backup_py(py_dir)
        write_yaml(args.__dict__, os.path.join(log_dir, "config.yaml"))

    writer = SummaryWriter(log_dir=log_dir)
    return writer, log_dir


def round_list(lst, dgts):
    return [round(val, dgts) for val in lst]

def calculate_principal_components(embeddings, num_components=3):
  """Calculates the principal components given the embedding features.
  Args:
    embeddings: A 2-D float tensor of shape `[num_pixels, embedding_dims]`.
    num_components: An integer indicates the number of principal
      components to return.
  Returns:
    A 2-D float tensor of shape `[num_pixels, num_components]`.
  """
  embeddings = embeddings - torch.mean(embeddings, 0, keepdim=True)
  _, _, v = torch.svd(embeddings)
  return v[:, :num_components] # Returns top num_components


def tsne_wrap(embeddings, num_components):
    shape = embeddings.shape
    embeddings = embeddings.view(-1, shape[-1])
    embeddings = TSNE(n_components=num_components, perplexity=30, n_iter=1000, verbose=True).fit_transform(embeddings)

    new_shape = list(shape[:-1]) + [num_components]
    embeddings = embeddings.reshape(new_shape)
    return embeddings


def pca(embeddings, num_components, principal_components=None):
    """Conducts principal component analysis on the embedding features.
    This function is used to reduce the dimensionality of the embedding.
    Args:
        embeddings: An N-D float tensor with shape with the 
        last dimension as `embedding_dim`.
        num_components: The number of principal components.
        principal_components: A 2-D float tensor used to convert the
        embedding features to PCA'ed space, also known as the U matrix
        from SVD. If not given, this function will calculate the
        principal_components given inputs.
    Returns:
        A N-D float tensor with the last dimension as  `num_components`.
    """
    shape = embeddings.shape
    embeddings = embeddings.view(-1, shape[-1])

    if principal_components is None:
        principal_components = calculate_principal_components(
            embeddings, num_components)
    embeddings = torch.mm(embeddings, principal_components)

    new_shape = list(shape[:-1]) + [num_components]
    embeddings = embeddings.view(new_shape)

    return embeddings, principal_components