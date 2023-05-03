"""
Script for training EBMs for discovering concepts, relations and operators.
"""
from copy import deepcopy
import itertools
import logging

import os
import sys
import pickle
import pprint as pp
import matplotlib.pyplot as plt
import matplotlib as mpl
logging.getLogger('matplotlib.font_manager').disabled = True
import time
from collections import defaultdict as ddict
from functools import lru_cache
import numpy as np
import networkx as nx
from tqdm import tqdm

import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from pytorch_net.util import Printer, to_nx_graph, draw_nx_graph, to_line_graph, nx_to_graph

from models.task_model import get_gnn_model
from data import get_dataloader
from loss import graph_task_loss, normalize_embedding
from args import get_args
from linear_eval import fine_tune_main
from kmeans import kmeans_with_initial_prototypes, kmeans, find_nearest_prototypes
from util import *
import networkx.algorithms.isomorphism as iso

# from reasoning.concept_env.BabyARC.code.dataset.dataset import * # Only needed for CLI generation

p = Printer()

def printTaskId2edgeDist(taskId2edgeDist):
    """Print the edge distribution for each task."""
    prnt_str = ""
    for taskId, edgeDist in enumerate(taskId2edgeDist):
        prnt_str += f"Task {taskId}  \n"
        for cnt, edgeDist in sorted( ((v,k) for k,v in edgeDist.items()), reverse=True):
            prnt_str += f"{edgeDist}: {cnt}, "
        prnt_str += "  \n"
    
    prnt_str = prnt_str.strip()
    print(prnt_str)
    return prnt_str

def main(args=None, dataloaders=None):
    """Main function for training and evaluating the model."""
    print(f"Using {torch.cuda.device_count()} GPUs")

    args = get_args(args)
    writer, log_dir = log_writer(args)

    torch.backends.cudnn.benchmark = True
    set_seed(args.random_seed)

    data_record = {"args": vars(args)}

    train_loader, train_loader_edge, val_loader, val_loader_edge, taskId2RelId = dataloaders if dataloaders else get_dataloader(args)

    model = get_gnn_model(args)
    model, device = model_parallel(model, args)

    edge_configs = get_edge_configs(args.num_rels, softmax=args.softmax_rel)

    task_loss_fn = graph_task_loss

    optimizer = optim.Adam(model.parameters(), 
        lr=args.learning_rate, weight_decay=args.weight_decay)
    
    scheduler = get_scheduler(args.lr_scheduler_type, optimizer, args.num_epochs)

    best_acc_save = False
    best_acc = 0
    start_epoch = 0
    start_step = 0
    loaded_model = False
    if args.load_latest and args.checkpoint_step != "best" :
        args.checkpoint_step = load_checkpoint_step(log_dir)

    if args.checkpoint_step == "best" or args.checkpoint_step > -1:
        state = load(log_dir, args.checkpoint_step, model, optimizer, scheduler=scheduler)
        best_acc = state.get("best_acc", 0)
        start_epoch = state["epoch"] + 1
        start_step = (state["step"] if isinstance(args.checkpoint_step, str) else args.checkpoint_step) + 1
        protos = state.get('protos', None)
        loaded_model = True
        if args.checkpoint_step == "best":
            print(f"best: acc {best_acc:.4f}, epoch {state['epoch']}, step {state['step']}")
    else:
        print('Checkpoint loading skipped.')

    if args.fine_tune:
        if not loaded_model:
            print("Warning, no model was loaded for linear fine-tuning")
        
        fine_tune_main(train_loader_edge, val_loader_edge, model, log_dir, args, device)
        return

    if args.evaluate:
        evaluate(val_loader, train_loader_edge, val_loader_edge, edge_configs, model, task_loss_fn, start_epoch, start_step, writer, data_record, args, device, args.padding_objs, args.total_tasks, protos)
        ipdb.set_trace()

    for epoch in range(start_epoch, args.num_epochs + 80000):
        val_dur = val_loss = val_acc = 0
        if epoch > 0 and epoch % args.print_interval2 == 0:
            val_dur, protos, accTr, accVal, taskAccTr, taskAccVal, edgeDistTrStr, edgeDistValStr = evaluate(val_loader, train_loader_edge, val_loader_edge, edge_configs, model, task_loss_fn, epoch, start_step, writer, data_record, args, device, args.padding_objs, args.total_tasks)

            if accVal > best_acc:
                best_acc = accVal
                best_acc_save = True

            writer.add_scalar('train/acc', accTr, start_step)
            writer.add_scalar('val/acc', accVal, start_step)
            writer.add_text('train/acc_list', f"{accTr} {taskAccTr}", start_step)
            writer.add_text('val/acc_list', f"{accVal} {taskAccVal}", start_step)
            writer.add_text('train/edge_dist', edgeDistTrStr, start_step)
            writer.add_text('val/edge_st', edgeDistValStr, start_step)
            writer.flush()

        start_step, tr_loss, tr_acc, train_dur = train(train_loader, model, task_loss_fn, 
                                        optimizer, epoch, start_step, 
                                        writer, data_record, args, device)
        print(f"Epoch: {epoch}\t T-dur {train_dur:.2f}\t T-loss {tr_loss:.3f}\tT-acc {tr_acc:.3f}\tV-dur {val_dur:.2f}\tV-loss {val_loss:.3f}\tV-acc {val_acc:.3f}")

        scheduler.step()

        if epoch > 0 and (best_acc_save or epoch % args.save_interval == 0 or epoch == args.num_epochs - 1):
            record_data(data_record, [epoch], ["save_epoch"])
            torch.cuda.empty_cache()
            pickle.dump(data_record, open(os.path.join(log_dir, args.pkl_name), "wb"))

            if epoch % args.save_interval == 0 or epoch == args.num_epochs - 1:
                save(log_dir, start_step, model, optimizer, scheduler, True, 
                    dict(epoch=epoch, step=start_step, best_acc=best_acc))
            if best_acc_save:
                best_acc_save = False
                print(f"Saving model with best val acc: {best_acc}")
                save(log_dir, "best", model, optimizer, scheduler, False, 
                    dict(epoch=epoch, step=start_step, best_acc=best_acc, protos=protos))


def train(loader, model, loss_fn, optimizer, epoch, start_step, writer, data_record, args, device):
    """Train the model for one epoch."""
    total_sum = acc_sum = loss_sum = 0
    start_time = time.time()
    model.train()

    for i, x in enumerate(loader, start=start_step):
        step_time = time.time()
        x = detach_to(x, device)
        edge_mask = x['edge_mask']

        out = model(x, softmax=args.softmax_rel, only_edge=False, aug=True, neg=True)

        loss, loss_d, pairwise_diff, block_diag = loss_fn(args, x, out, step=i, \
                        alphas=model.module.gin_task.ex2alpha(x['ex_ids']) if args.is_lookup_mask else None)
        loss_sum += loss * edge_mask.shape[0]
        total_sum += edge_mask.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.is_lookup_mask:
            model.module.gin_task.ex2alpha.weight.data = model.module.gin_task.ex2alpha.weight.data.clip(0.0, 1.0)

        # Logging and Visualization
        log_record(loss_d, "train", epoch, i, writer, data_record)

        vis_edge_feat = out['edge'] * edge_mask[:, :, None]
        if (i - start_step) % args.print_interval == 0:
            print(f"Epoch {epoch}  step: {i}\t loss: {loss_d['loss']:.4f}\tintra: {loss_d['intra_loss']:.4f}\tinter: {loss_d['inter_loss']:.4f}\tdur: {time.time() - step_time:.2f}s/it\tlr: {optimizer.param_groups[0]['lr']:.3e}")

            # print(model.module.gin_task.inner_lr.item(), model.module.gin_task.save_il_print)
            if args.is_lookup_mask:
                print(model.module.gin_task.ex2alpha.weight[:6])

            if args.write_tb_imgs:
                vis_pairwise(pairwise_diff, writer, i)
                vis_blockdiag(1 - block_diag, writer, i)
                vis_edgefeat(vis_edge_feat, writer, i)
            writer.flush()

        if epoch % args.print_interval3 == 0 and i == start_step:
            print("pairwise diff")
            print(pairwise_diff)
            print("-" * 20)
            print("edge feat")
            print(vis_edge_feat.detach().cpu().numpy().round(2))
            print("-" * 20)
            print("gin output")
            print(out["gin_output"])
            print("gin output_sum")
            print(out["gin_output_sum"])

    return i + 1, loss_sum/total_sum, 0, time.time() - start_time

def validate(loader, model, loss_fn, epoch, start_step, writer, data_record, args, device):
    """Validate the model on the validation set."""
    model.eval()
    start_time = time.time()
    
    batch_d = ddict(float)
    cnt = 0       
    
    with torch.no_grad(): # can't do nograd for maml style - actually just reenable with torch.enable_grad()
        for i, x in enumerate(loader, start=start_step):
            x = detach_to(x, device)
            task_ids = x['task_ids']
            
            out = model(x, softmax=args.softmax_rel, only_edge=False, aug=False, neg=True)

            loss, loss_d, _, _ = loss_fn(args, x, out, step=i)

            for k, v in loss_d.items():
                batch_d[k] += v * task_ids.shape[0] # Because it's per-task loss
            cnt += task_ids.shape[0]
                
    for k in batch_d.keys():
        batch_d[k] /= cnt

    # Logging
    log_record(batch_d, "val", epoch, start_step, writer, data_record)
    
    return time.time() - start_time

def compute_best_config(edge_configs, relId2edgeDist):
    start_time = time.time()
    
    total_edges = 0 # Could be cached but it also takes like 0.00x sec
    for rel_id, edge_dist in enumerate(relId2edgeDist):
        if rel_id == 0:
            continue
        count = 0
        for _, cnt in edge_dist.items():
            total_edges += cnt
        # rel_total_cnt.append(count)
    
    best_config = None
    best_acc = 0
    for edge_config in edge_configs:
        correct_edges = 0
        for rel_id, gt_edge in enumerate(edge_config):
            if rel_id != 0:
                correct_edges += relId2edgeDist[rel_id][tuple(gt_edge)]
        acc = correct_edges / total_edges
        if acc > best_acc:
            best_acc = acc
            best_config = edge_config

    return best_config

def evaluate(val_loader, train_loader_edge, val_loader_edge, edge_configs, model, task_loss_fn, epoch, start_step, writer, data_record, args, device, padding_objs, total_tasks, protos=None):
    """Evaluate the model on the validation set."""
    val_dur = validate(val_loader, model, task_loss_fn, epoch, start_step, writer, data_record, args, device)
    relTaskDistTr, taskDistTr, relDistTr, protos, labelsTr, edgeMaskTr, taskIdsTr = edge_dist_fast(train_loader_edge, model, args, device, False, protos=protos)
    relTaskDistVal, taskDistVal, relDistAll, _, labelsVal, edgeMaskVal, taskIdsVal = edge_dist_fast(val_loader_edge, model, args, device, False, relDistTr, protos)

    if args.evaluate and args.compute_mcs:
        graphsTr = compute_graphs(labelsTr, edgeMaskTr, padding_objs)
        graphsVal = compute_graphs(labelsVal, edgeMaskVal, padding_objs)
        mcsTr = compute_mcs(graphsTr, taskIdsTr, total_tasks)

    best_config = compute_best_config(edge_configs, relDistAll)
    accTr, taskAccTr = edge_acc_fast(relTaskDistTr, best_config)
    accVal, taskAccVal = edge_acc_fast(relTaskDistVal, best_config)
    
    taskAccTr, taskAccVal = round_list(taskAccTr, 4), round_list(taskAccVal, 4)

    print(f"Train edge acc: {accTr:.4f} \ttask acc: {taskAccTr} \tbest_config: {best_config}")
    print(f"Val edge acc: {accVal:.4f} \ttask acc: {taskAccVal}")
    p.print("Train edge dist", banner_size=70)
    edgeDistTrStr = printTaskId2edgeDist(taskDistTr)
    p.print("Val edge dist:", banner_size=70)
    edgeDistValStr = printTaskId2edgeDist(taskDistVal)

    return val_dur, protos, accTr, accVal, taskAccTr, taskAccVal, edgeDistTrStr, edgeDistValStr

def g_helper(g_lst, return_lst=False):
    """Helper function to convert g_lst to nx graph"""
    g_lg = to_line_graph(g_lst)
    g = get_nx_graph(g_lg, is_directed=False)
    
    if return_lst:
        return g_lg, g
    return g

def get_mcs(g1_inpt, g2_inpt, convert_g1=True, convert_g2=True):
    """Get the Maximum Common Subgraph (MCS) between two graphs"""
    g1_lst, g1 = g_helper(g1_inpt, True) if convert_g1 else g1_inpt
    g2_lst, g2 = g_helper(g2_inpt, True) if convert_g2 else g2_inpt
    
    def node_match(node_dict1, node_dict2):
        return node_dict1["type"] == node_dict2["type"]

    ismags = nx.isomorphism.ISMAGS(g1, g2, node_match=node_match)
    mcs_base = list(ismags.largest_common_subgraph(symmetry=False))
    mcs = [g for g in mcs_base if len(g) > 1]

    if len(mcs) == 0:
        # Due to imperfect prediction, sometimes may not have an MCS
        return None
    
    if g1.number_of_edges() < g2.number_of_edges():
        g_lst_smaller = g1_lst
        g_mcs = g1.subgraph(list(mcs[0].keys()))
    else:
        g_lst_smaller = g2_lst
        g_mcs = g2.subgraph(list(mcs[0].values()))

    g_lst_node_dct = {node[0] : node[1] for node in g_lst_smaller if type(node[0]) is str and node[0] in g_mcs}
    mcs_lst = []
    for item in g_lst_smaller:
        if type(item[0]) is tuple:
            if item[0][0] in g_lst_node_dct and item[0][1] in g_lst_node_dct:
                mcs_lst.append(item)
    mcs_lst += list(g_lst_node_dct.items())

    return mcs_lst, g_mcs


def get_group_mcs(tid_buffer, tid=None):
    """Get the MCS for a group of graphs"""
    mcs_lst = tid_buffer[0]
    mcs_lg, g_mcs = g_helper(mcs_lst, True)
    if tid == 2:
        pass
    for g_lst in tid_buffer[1:]:
        res = get_mcs((mcs_lg, g_mcs), g_lst, convert_g1=False)
        if res is not None:
            mcs_lg, g_mcs = res
    return mcs_lg, g_mcs


def compute_mcs(graphs, task_ids, total_tasks):
    """Compute the MCS for each task"""
    task_ids = task_ids.tolist()
    task_graph_buffer = [[] for _ in range(total_tasks)]
    task_graph_step1 = [[] for _ in range(total_tasks)]
    task_graph_stats = [[] for _ in range(total_tasks)]

    GROUP_SIZE = 5
    t1 = time.time()
    for graph, task_id in tqdm(zip(graphs, task_ids)):
        tid_buffer = task_graph_buffer[task_id]
        tid_buffer.append(graph)
        
        if len(tid_buffer) >= GROUP_SIZE:
            group_mcs = get_group_mcs(tid_buffer, task_id)
            task_graph_step1[task_id].append(group_mcs)
            tid_buffer.clear()
    

    for task_id, tid_buffer in enumerate(task_graph_buffer):
        if len(tid_buffer) > 0:
            group_mcs = get_group_mcs(tid_buffer, task_id)
            task_graph_step1[task_id].append(group_mcs)
            tid_buffer.clear()

    print("Subgroup MCS time", time.time() - t1)
    t1 = time.time()

    
    for task_id, group_mcs_list in tqdm(enumerate(task_graph_step1)):
        tid_stats = task_graph_stats[task_id]
        for group_mcs in group_mcs_list:
            mcs_lst, g_mcs = group_mcs
            seen = False
            for unique_g in tid_stats:
                g_unique = unique_g[1]
                if nx.is_isomorphic(g_unique, g_mcs):
                    seen = True
                    unique_g[2] += 1
                    break
            if not seen:
                tid_stats.append([mcs_lst, g_mcs, 1])
    
    print("Isomorphism count", time.time() - t1)

    top_n = 3
    print("Group size:", GROUP_SIZE)
    for task_id, tid_stats in enumerate(task_graph_stats):
        print(f"ID {task_id}:")
        for stat in sorted(tid_stats, key = lambda x: x[2], reverse=True)[:top_n]:
            print(f"Count: {stat[2]}, MCS: {pretty_print_stat(from_line_graph(stat[0]))}")

    return task_graph_stats

def pretty_print_stat(stat):
    """Pretty print the MCS"""
    stat_str = ""
    for elem in stat:
        if type(elem[0]) is int:
            continue
        elem_pretty = [elem[0], str(elem[1])]
        stat_str += str(elem_pretty) + ", "
    return stat_str[:-2]

def from_line_graph(lg):
    """Convert the line graph to the original graph"""
    g_lst = []
    max_id = 0
    for item in lg:
        if type(item[0]) is str:
            n1, n2 = list(map(int, item[0].split(",")))
            max_id = max(max_id, n1, n2)
            g_lst.append([(n1, n2), item[1]])
    for i in range(max_id + 1):
        g_lst.append([i, ""])
    return g_lst

def compute_graphs(labels, edge_masks, padding_objs):
    """Compute the graphs from the edge masks and labels"""
    triu_indices = torch.triu_indices(padding_objs, padding_objs, offset=1)
    rows, cols = (triu_indices[0].tolist(), triu_indices[1].tolist())
    edge_masks = edge_masks.cpu().numpy()
    labels = labels.tolist()
    label_idx = 0
    graphs = []
    a = time.time()
    #[[(0, 1), 'SameColor'], [0, ''], [1, ''], [(1, 0), 'SameColor']]
    for ex_id, masks in enumerate(edge_masks):
        graph = []
        pad_labels = []

        # if ex_id == 2:
        #     ipdb.set_trace()
        for mask in masks:
            if mask:
                pad_labels.append(labels[label_idx])
                label_idx += 1
            else:
                pad_labels.append(-1)
        max_id = -1
        for idx, (r, c, mask) in enumerate(zip(rows, cols, masks)):
            if mask:
                graph.append([(r, c), pad_labels[idx]]) #str()
                max_id = max(r, c, max_id)
        for idx in range(0, max_id + 1):
            graph.append([idx, ''])
        graphs.append(graph)

    print(edge_masks.sum(), len(labels), label_idx, time.time() - a)
    return graphs

def edge_dist_fast(loader, model, args, device, bw_01, relId2edgeDist=None, protos=None):
    """Compute the edge distance for each task"""
    model.eval()
    start_time = time.time()
    
    numRels = args.num_rels
    relTaskId2edgeDist = [[ddict(int) for _ in range(numRels + 1)] for _ in range(args.total_tasks)]
    taskId2edgeDist = [ddict(int) for _ in range(args.total_tasks)]
    if relId2edgeDist is None:
        relId2edgeDist = [ddict(int) for _ in range(numRels + 1)]

    lazy_edges = torch.eye(3, device=device, dtype=torch.int32)

    with torch.no_grad():
        for i, x in enumerate(loader):
            x = detach_to(x, device)
            task_ids = x['task_ids']
            edge_mask = x['edge_mask']
            gt_edge = x['gt_edge']

            out = model(x, softmax=args.softmax_rel, only_edge=not args.show_tsne_task, aug=False, neg=False) #[B, 3, 3] and [B, 3]
            edge = out['edge']
            
            if edge.size(-1) > 3:
                if protos is None:
                    num_protos = args.num_rels
                    labels, protos = kmeans(edge[edge_mask.bool()], iterations=15, num_protos=num_protos, return_proto=True)
                else:
                    num_protos = args.num_rels
                    labels, protos = kmeans_with_initial_prototypes(edge[edge_mask.bool()], protos, iterations=10, num_protos=num_protos, return_proto=True)
                    
                rel_matrix_s_mask = lazy_edges[labels]
            elif not bw_01:
                edge = (edge + 1.0)/2.0
                rel_matrix_s_mask = edge[edge_mask.bool()]
                rel_matrix_s_mask = rel_matrix_s_mask.round().int()

            if args.evaluate and args.show_tsne:
                # Visualize
                rel_matrix_mask = edge[edge_mask.bool()]
                subset_rel = torch.randperm(rel_matrix_mask.shape[0])[:1500]
                rel_matrix_s_np = tsne_wrap(rel_matrix_mask[subset_rel], 2)
                e_labels = gt_edge[edge_mask.bool()][subset_rel]
                np_mat = rel_matrix_s_np
                vis_edge_tsne(np_mat, e_labels)
                vis_edge_tsne(np_mat, labels[subset_rel])

            if args.evaluate and args.show_tsne_task:
                # Visualize
                task_matrix = out['gin_output_sum']
                task_matrix = normalize_embedding(task_matrix)
                subset_rel = torch.randperm(task_matrix.shape[0])[:1500]
                task_matrix_s_np = tsne_wrap(task_matrix[subset_rel], 2)
                e_labels = task_ids[subset_rel]
                np_mat = task_matrix_s_np
                vis_edge_tsne(np_mat, e_labels, args.total_tasks)
                if args.classify:
                    cl_labels = out['task_id_pred'][subset_rel]
                    vis_edge_tsne(np_mat, cl_labels, args.total_tasks)
                    task_acc = torch.eq(task_ids, out['task_id_pred']).float().mean()
                    print(f"Acc of task_id {task_acc}")
            
            edge_mask_b = edge_mask.bool()
            gt_edge_mask = gt_edge[edge_mask_b].unsqueeze(-1)
            task_ids_mask = task_ids.unsqueeze(1).expand(edge_mask.shape)[edge_mask_b].unsqueeze(-1)
            
            uniq_edges_comp, counts = torch.unique(edge_matrix_comp, return_counts=True, dim=0)
            for uniq_edge_comp, count in zip(uniq_edges_comp.tolist(), counts.tolist()):
                edge = tuple(uniq_edge_comp[:3])
                task_id, rel_id = uniq_edge_comp[3:]
                relTaskId2edgeDist[task_id][rel_id][edge] += count

    # Quick code to compute aggr values
    for task_id in range(args.total_tasks):
        for rel_id in range(numRels + 1):
            for edge, cnt in relTaskId2edgeDist[task_id][rel_id].items():
                taskId2edgeDist[task_id][edge] += cnt    
                relId2edgeDist[rel_id][edge] += cnt

    print(f"{time.time() - start_time :.2f}")
    return relTaskId2edgeDist, taskId2edgeDist, relId2edgeDist, protos, labels, edge_mask_b, task_ids

def edge_acc_fast(relTaskId2edgeDist, best_config):
    """Compute the edge accuracy for each task"""
    taskAcc = []
    total_num_edges = 0
    total_correct_edges = 0
    for task_id_dist in relTaskId2edgeDist:
        correct_edges = 0
        num_edges = 0
        for rel_id, rel_id_dist in enumerate(task_id_dist):
            if rel_id == 0: 
                continue
            edge = best_config[rel_id]
            correct_edges += rel_id_dist[edge]
            for edge, cnt in rel_id_dist.items():
                num_edges += cnt
        total_num_edges += num_edges
        total_correct_edges += correct_edges
        taskAcc.append(correct_edges/num_edges)

    acc = total_correct_edges/total_num_edges
    return acc, taskAcc

def vis_pairwise(pairwise, writer, i):
    """Visualize the pairwise difference"""
    pairwise = pairwise.clip(max=5)
    pairwise = pairwise / 5
    writer.add_image('pairwise_diff', pairwise, i, dataformats='WC')

def vis_blockdiag(blockdiag, writer, i):
    """Visualize the block diagonal matrix"""  
    writer.add_image('blockdiag', blockdiag, i, dataformats='WC')
    
def vis_numobjs(num_objs, writer, i):
    """Visualize the number of objects"""
    block = torch.eq(num_objs[None], num_objs[:, None])
    writer.add_image('num_objs_viz', block, i, dataformats='WC')
    
def vis_edgefeat(edgefeat, writer, i):
    """Visualize the edge features"""
    edgefeat = edgefeat.sum(dim=1)
    writer.add_image('edgefeat', edgefeat, i, dataformats='WC')

def vis_edge_tsne(edges_np, e_labels, max_label=4):
    """Visualize the edge features using t-SNE"""
    
    fig = plt.figure()
    ax = fig.add_subplot()
    e_labels = e_labels.cpu().numpy()
    for i in range(max_label):
        pts = edges_np[e_labels == i]
        ax.scatter(pts[:, 0], pts[:, 1], s=3.0, label=i)
    # ax.scatter(edges_np[:, 0], edges_np[:, 1], s=3.0, c=e_labels.tolist())
    ax.legend()
    plt.show()

def edge_acc(loader, model, writer, edge_configs, data_record, args, device, bw_01, ignore_none=True):
    """Compute the edge accuracy for each task"""
    start_time = time.time()
    model.eval()
    
    best_acc = 0
    best_config = None
    taskIdBestAcc = [0] * args.total_tasks
    taskIdPreds = [[] for _ in range(args.total_tasks)]
    taskIdGT = [[] for _ in range(args.total_tasks)]

    with torch.no_grad():
        for i, (mask_imgs, gt_edges, edge_mask, task_ids, mask_imgs_rand, edge_mask_rand) in enumerate(loader):
            mask_imgs = mask_imgs.detach().to(device)
            task_ids = task_ids.detach().to(device)
            edge_mask = edge_mask.detach().to(device)
            
            rel_matrix_s, _, _ = model(mask_imgs, None, edge_mask, args.softmax_rel, True)
            
            if rel_matrix_s.size(-1) > 3:
                rel_matrix_s = rel_matrix_s[:, :, :3]
            
            if not bw_01:
                rel_matrix_s = (rel_matrix_s + 1.0)/2.0
            
            for idx, (edges, gt_edge, task_id, mask) in enumerate(zip(rel_matrix_s.round().int(), gt_edges, task_ids, edge_mask)):
                task_id = task_id.item()
                task_edges = [tuple(edge.tolist()) for edge in edges[mask.bool()]]
                taskIdPreds[task_id].extend(task_edges)    
                task_gt = [gt.item() for gt in gt_edge[mask.bool()]]
                taskIdGT[task_id].extend(task_gt)
    
    for edge_config in edge_configs:
        taskIdAccSum = [0] * args.total_tasks
        taskIdTotalEdge = [0] * args.total_tasks
        total_acc_sum = 0
        total_edges = 0
        
        for taskId in range(args.total_tasks):
            edge_preds = taskIdPreds[taskId]
            edge_GT = taskIdGT[taskId]
            
            # Per example output is flattened
            for edge, label in zip(edge_preds, edge_GT):
                if label == 0 and ignore_none:
                    continue

                total_edges += 1
                taskIdTotalEdge[taskId] += 1

                if edge == edge_config[label]:
                    taskIdAccSum[taskId] += 1
                    total_acc_sum += 1
                        
        config_acc = total_acc_sum/total_edges
        if config_acc > best_acc:
            best_acc = config_acc
            best_config = edge_config
            taskIdBestAcc = [p/q for p, q in zip(taskIdAccSum, taskIdTotalEdge)]
                    
    print(f"{time.time() - start_time :.2f}")
    return best_acc, taskIdBestAcc, best_config

def edge_dist(loader, model, writer, data_record, args, device, bw_01):
    """Compute the edge distribution for each task"""
    model.eval()
    start_time = time.time()
    
    taskId2edgeDist = [ddict(int) for _ in range(args.total_tasks)]
    with torch.no_grad():
        for i, (mask_imgs, gt_edges, edge_mask, task_ids, mask_imgs_rand, edge_mask_rand) in enumerate(loader):
            mask_imgs = mask_imgs.detach().to(device)
            task_ids = task_ids.detach().to(device)
            edge_mask = edge_mask.detach().to(device)

            rel_matrix_s, _, _ = model(mask_imgs, None, edge_mask, args.softmax_rel, True) #[252, 3, 3] and [252, 3]
            
            if rel_matrix_s.size(-1) > 3:
                rel_matrix_s = rel_matrix_s[:, :, :3]

            if not bw_01:
                rel_matrix_s = (rel_matrix_s + 1.0)/2.0
            
            for idx, (edges, task_id, mask) in enumerate(zip(rel_matrix_s.round().int(), task_ids, edge_mask)):
                for edge, valid in zip(edges, mask):
                    if valid.item():
                        edge_tup = tuple(edge.tolist())
                        taskId2edgeDist[task_id.item()][edge_tup] += 1
    print(f"{time.time() - start_time :.2f}")
    return taskId2edgeDist

def edge_dist_sup(loader, model, writer, data_record, args, device):
    """Compute the edge distribution for each task using supervision"""
    model.eval()
    start_time = time.time()
    
    taskId2edgeDist = [ddict(int) for _ in range(args.total_tasks)]
    with torch.no_grad():
        for i, (masked_imgs, gt_edges, edge_mask, task_ids) in enumerate(loader):
            masked_imgs = masked_imgs.detach().to(device)
            gt_edges = gt_edges.detach().to(device)
            task_ids = task_ids.detach().to(device)

            edge_feature_matrix_s, node_feats = model(masked_imgs, edge_mask, args.softmax_rel, True)
            
            for idx, (edges, task_id, mask) in enumerate(zip(edge_feature_matrix_s.round(), task_ids, edge_mask)):
                for edge, valid in zip(edges, mask):
                    if valid.item():
                        edge_tup = tuple(edge.int().tolist())
                        taskId2edgeDist[task_id.item()][edge_tup] += 1
    print(f"{time.time() - start_time :.2f}")
    return taskId2edgeDist

def train_sup(loader, model, loss_fn, optimizer, epoch, start_step, writer, data_record, args, device):
    """Train the model using supervision, for only sanity check"""
    total_sum = acc_sum = loss_sum = 0
    start_time = time.time()
    model.train()
    for i, (masked_imgs, gt_edges, edge_mask, task_ids) in enumerate(loader, start=start_step):
        masked_imgs = masked_imgs.detach().to(device)
        gt_edges = gt_edges.detach().to(device)
        edge_mask = edge_mask.detach().to(device)
        step_time = time.time()
        
        edge_feature_matrix_s, node_feats = model(masked_imgs, edge_mask, args.softmax_rel, True)
         
        edge_mask_ignore_none = torch.ne(gt_edges.flatten(0, 1), 0) * edge_mask.flatten(0, 1)
        
        loss_masked = loss_fn(edge_feature_matrix_s.flatten(0, 1), gt_edges.flatten(0, 1)) * edge_mask_ignore_none
        
        acc_masked = torch.eq(edge_feature_matrix_s.flatten(0, 1).argmax(dim=-1), gt_edges.flatten(0, 1)) * edge_mask_ignore_none
        
        loss_mask_sum = loss_masked.sum()
        sum_curr = edge_mask_ignore_none.sum().detach()
        
        loss = loss_masked.mean() #loss_mask_sum/sum_curr
        total_sum += sum_curr
        loss_sum += loss_mask_sum
        acc_sum += acc_masked.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging and Visualization
        writer.add_scalar('train/loss', loss.item(), i)
        record_data(data_record, [epoch, i, "train", loss],
                        ["epoch", "step", "type", "loss"])
        
        vis_edge_feat = edge_feature_matrix_s * edge_mask[:, :, None]
        if i % args.print_interval == 0 or i == start_step:
            print(f'Epoch: {epoch} step: {i}', loss.item(), f"dur: {time.time() - step_time:.2f}s/it")
            if args.write_tb_imgs:
                vis_pairwise(pairwise_diff, writer, i)
                vis_blockdiag(1 - block_diag, writer, i)
                vis_edgefeat(vis_edge_feat, writer, i)
            writer.flush()
        
        if epoch % args.print_interval2 == 0 and i == start_step:
            print("edge feat")
            print(vis_edge_feat.detach().cpu().numpy().round(2))
    
    writer.add_scalar('train/acc', acc_sum/total_sum, i)
    record_data(data_record, [epoch, i, "train", acc_sum/total_sum], ["epoch", "step", "type", "acc"])
    return i + 1, loss_sum/total_sum, acc_sum/total_sum, time.time() - start_time            

def validate_sup(loader, model, loss_fn, epoch, start_step, writer, data_record, args, device):
    """Validate the model using supervision, for only sanity check"""
    total_sum = acc_sum = loss_sum = 0
    start_time = time.time()
    model.eval()
    
    with torch.no_grad():
        for i, (masked_imgs, gt_edges, edge_mask, task_ids) in enumerate(loader, start=start_step):
            masked_imgs = masked_imgs.detach().to(device)
            gt_edges = gt_edges.detach().to(device)
            edge_mask = edge_mask.detach().to(device)
            step_time = time.time()

            edge_feature_matrix_s, node_feats = model(masked_imgs, edge_mask, args.softmax_rel, True)

            edge_mask_ignore_none = torch.ne(gt_edges.flatten(0, 1), 0) * edge_mask.flatten(0, 1)
            
            loss_masked = loss_fn(edge_feature_matrix_s.flatten(0, 1), gt_edges.flatten(0, 1)) * edge_mask_ignore_none

            acc_masked = torch.eq(edge_feature_matrix_s.flatten(0, 1).argmax(dim=-1), gt_edges.flatten(0, 1)) * edge_mask_ignore_none
            
            loss_mask_sum = loss_masked.sum()
            sum_curr = edge_mask_ignore_none.sum()
            
            loss = loss_mask_sum/sum_curr
            total_sum += sum_curr
            loss_sum += loss_mask_sum
            acc_sum += acc_masked.sum()


            # Logging and Visualization
            writer.add_scalar('val/loss', loss.item(), i)
            record_data(data_record, [epoch, i, "val", loss.item()],
                            ["epoch", "step", "type", "loss"])
    
    writer.add_scalar('val/acc', acc_sum/total_sum, i)
    record_data(data_record, [epoch, i, "val", acc_sum/total_sum], ["epoch", "step", "type", "acc"])
    record_data(data_record, [epoch, i, "val", loss_sum/total_sum], ["epoch", "step", "type", "epoch_loss"])
    return loss_sum/total_sum, acc_sum/total_sum, time.time() - start_time 

if __name__ == "__main__":
    main()
    