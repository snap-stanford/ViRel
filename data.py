import random
import time
import ipdb
from torch.nn import functional as F
import torch
from torch import nn
from tqdm import tqdm
import os
import pickle as pkl

from pytorch_net.util import init_args, get_graph_edit_distance
from concept_energy import get_dataset, transform_pos_data
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset
from functools import lru_cache
from collections import defaultdict as ddict

from util import get_num_edges, load_pkl

AGGR_TASK_CACHE = "cache"

EXAMPLES_PER_TASK = 6

def aggregate_tasks(dataset, dataset_fname=None, seen_graphs=None, max_count=None):
    # dataset[0][x][3]['obj_spec_core']
    # all the same for x
    def process_rel(rel):
        ids = rel[0]
        new_ids = (int(ids[0][4:]), int(ids[1][4:]))
        return [new_ids, rel[1]]
    
    def get_rel_graph(relations):
        rel_graph = [process_rel(rel_s) for rel_s in rel if rel_s[1] != 'Attr']
        max_id = 0
        rel_names = set()
        mirrored_edges = []
        for ids, name in rel_graph:
            max_id = max(max_id, max(ids))
            rel_names.add(name)
            if "Same" in name: # Undirected edges
                mirrored_edges.append([tuple(reversed(ids)), name]) 
        for idx in range(max_id + 1):
            rel_graph.append([idx, ""])
        return rel_graph + mirrored_edges, rel_names, max_id + 1

    if seen_graphs is None:
        seen_graphs = []
    
    example_idx = 0
    taskId2RelId = []
    relName2Id = {"None": 0}
    task_ids = []
    tasks_per_taskid = ddict(int)
    filtered_idx = []
    ds_stats = ddict(int)

    idxTask2TaskId = []
    saveCache = True
    if dataset_fname:
        idxTask2TaskId, seen_graphs, taskId2RelId = load_pkl(os.path.join(AGGR_TASK_CACHE, dataset_fname), ([], [], []))
        if len(seen_graphs) > 0:
            print(f"Using cached for aggregate_tasks {dataset_fname}")
            saveCache = False
    
    # Store task_id at [3]['graph_task_id'] in the example
    for idx_task, task in enumerate(dataset):
        rel = task[0][3]['obj_spec_core']
        rel_graph, rel_names, num_objs_core = get_rel_graph(rel)
        #num_objs_core = len(task[0][3]['obj_masks']) # May include distractors now
        
        for name in rel_names:
            if name not in relName2Id:
                relName2Id[name] = len(relName2Id)

        seen = False
        if saveCache:
            task_id = len(seen_graphs)
            for idx, seen_graph in seen_graphs:
                if get_graph_edit_distance(seen_graph, rel_graph) == 0:
                    seen = True
                    task_id = idx
                    break
            idxTask2TaskId.append(task_id)
        else:
            seen = True
            task_id = idxTask2TaskId[idx_task]
        tasks_per_taskid[task_id] += 1
            
        if max_count and tasks_per_taskid[task_id] <= max_count:
            filtered_idx.append(idx_task)
            task_ids.append(task_id)

        if (max_count and tasks_per_taskid[task_id] <= max_count) or not max_count:
            
            for example in task:
                example[3]['task_id'] = task_id
                example[3]['example_id'] = example_idx
                example_idx += 1
                num_objs = len(example[3]['obj_masks'])
                ds_stats[(num_objs_core, num_objs-num_objs_core)] += 1
        
        if not seen:
            relIds = [relName2Id[name] for name in rel_names]
            relIds += [0] * (get_num_edges(num_objs_core) - len(relIds))
            taskId2RelId.append(relIds)
            seen_graphs.append((task_id, rel_graph))
    
    if saveCache:
        with open(os.path.join(AGGR_TASK_CACHE, dataset_fname), 'wb') as f:
            cache = (idxTask2TaskId, seen_graphs, taskId2RelId)
            pkl.dump(cache, f)

    for idx, seen_graph in seen_graphs:
        print(f"ID: {idx} . Num Tasks: {tasks_per_taskid[idx]}", end="")
        print(f"  . Filtered Tasks: {min(tasks_per_taskid[idx], max_count)}" if max_count else "")
        print(seen_graph)
        
    if max_count: # Filter the dataset
        for task_id in range(len(taskId2RelId)):
            if tasks_per_taskid[task_id] < max_count:
                # Assuming that every task_id has at least max_count
                print(f"Warning, taskid {task_id} does not have up to {max_count} samples")

        dataset_filtered = dataset[:]
        dataset_filtered.idx_list = [dataset.idx_list[idx] for idx in filtered_idx]
        dataset = dataset_filtered
    
    return dataset, torch.LongTensor(task_ids), seen_graphs, taskId2RelId, relName2Id, example_idx, ds_stats

def create_collate_fn(relName2Id, num_augs, padding_objs, augmentation=True, shuffle_masks=True):
    # Store the taskID 2 rel mapping
   
    triu_indices = torch.triu_indices(padding_objs, padding_objs, offset=1)
    rows, cols = (triu_indices[0].tolist(), triu_indices[1].tolist())
    rc_shuffle_idx = {} # determines the idx of original "edge location" after shuffling
    for idx, (r, c) in enumerate(zip(rows, cols)):
        rc_shuffle_idx[(r, c)] = idx
        rc_shuffle_idx[(c, r)] = idx

    def mask_collate_fn(batch):
        mask_img_embeds = []
        mask_img_aug_embeds = []
        gt_edge = []
        objs_mask = []
        edge_mask = []
        task_ids = []
        example_ids = []
        shuffle_rel_idx = []
        
        # masked_img_embed_neg_lst = []
        for batch_id, task in enumerate(batch):
            task_id = task[0][3]['task_id']
            for batch_ex_id, example in enumerate(task):
                # masks_v = [example[3]['obj_masks'][f'obj_{i}'][None, None] for i in range(len(example[3]['obj_masks']))] #tuple of num_obj * [1 x 1 x w x h]
                masks_kv = list(example[3]['obj_masks'].items())
                relIds = []
                relShuffleIdx = []
                rel_spec_dict = {k : v for k, v in example[3]['obj_spec_core'] if v != "Attr"}
                rel_spec_dict.update({tuple(reversed(k)) : v for k, v in example[3]['obj_spec_core'] if v != "Attr"})
                
                num_obj = len(masks_kv)
                num_pads = padding_objs - num_obj
                masks_kv_idx = list(range(num_obj))
                if shuffle_masks: 
                    masks_kv_idx = random.sample(masks_kv_idx, k=len(masks_kv))
                    masks_kv = [masks_kv[idx] for idx in masks_kv_idx]
                masks_kv_idx += list(range(num_obj, padding_objs)) # Add in the indexes for padding objs (null objs)
                    
                for r, c in zip(rows, cols):
                    if r < num_obj and c < num_obj:
                        find_rel = (masks_kv[r][0], masks_kv[c][0])
                        relIds.append(relName2Id[rel_spec_dict.get(find_rel, "None")])    
                    else:
                        relIds.append(0)
                    relShuffleIdx.append(rc_shuffle_idx[masks_kv_idx[r], masks_kv_idx[c]])
                
                masks_v = [mask[1][None, None] for mask in masks_kv]
                img = example[0][0][None] # [1 x num_color x w x h]
                
                objs_mask.append([1] * num_obj + [0] * num_pads)
                # edge_mask.append(get_valid_edges(num_pads)) #Sanity check
                
                gt_edge.append(torch.LongTensor(relIds))
                task_ids.append(task_id)
                example_ids.append(example[3]['example_id'])
                shuffle_rel_idx.append(torch.LongTensor(relShuffleIdx))
                
                masked_img_augs = []
                for aug_idx in range(num_augs):
                    masks_v_curr = masks_v
                    img_curr = img
                    if augmentation or aug_idx > 0:
                        augment_prob = 0.9 if aug_idx == 0 else 1.0
                        img_curr, masks_v_curr, _, _ = transform_pos_data((img_curr, masks_v_curr, None, None), \
                                                                     f"color+flip+rotate+resize:{augment_prob}", None)
                        
                    img_curr = img_curr[:, 1:] #Remove the background color dimension
                    masks = torch.stack(masks_v_curr).squeeze(1) #Remove batch dim # [num_obj x 1 x w x h]
                    masked_img = img_curr * masks # [num_obj x num_color x w x h]
                    
                    # need to pad first dim
                    masked_img = F.pad(masked_img, (0, 0, 0, 0, 0, 0, 0, num_pads), 'constant', 0)
                    
                    if aug_idx == 0:
                        mask_img_embeds.append(masked_img)
                    else:
                        masked_img_augs.append(masked_img)
                        
                if num_augs > 1:
                    mask_img_aug_embeds.append(torch.stack(masked_img_augs))
        
        mask_img_embeds_pt = torch.stack(mask_img_embeds)
        mask_img_aug_embeds_pt = torch.stack(mask_img_aug_embeds) if num_augs > 1 else None
        objs_mask_pt = torch.tensor(objs_mask)
        edge_mask_pt = torch.bitwise_and(objs_mask_pt[:, rows], objs_mask_pt[:, cols])

        mask_img_embeds_rand = mask_img_embeds_pt
        mask_img_embeds_rand_shp = mask_img_embeds_rand.shape
        
        mask_img_embeds_rand = mask_img_embeds_rand.flatten(0, 1)
        mask_img_embeds_rand_perm = torch.randperm(mask_img_embeds_rand.shape[0])
        
        mask_img_embeds_rand = mask_img_embeds_rand[mask_img_embeds_rand_perm]
        mask_img_embeds_rand = mask_img_embeds_rand.view(mask_img_embeds_rand_shp)
        
        objs_mask_rand_pt = objs_mask_pt.flatten(0, 1)[mask_img_embeds_rand_perm].view(objs_mask_pt.shape)
        edge_mask_rand_pt = torch.bitwise_and(objs_mask_rand_pt[:, rows], objs_mask_rand_pt[:, cols])

        # mask_img : BS * 6, num_pad_objs, w, h
        # mask_img_augs : BS * 6, num_augs, num_pad_objs, w, h
        # mask_img_neg : BS * 6, num_neg, num_pad_objs, w, h
        # gt_edge : BS * 6, 3 - where 3 is GT shuffled relation of each edge in TRIU
        # GCN_mask : BS * 6, 3 - where each 3 is 1,1,1 or 1,0,0 depending on 3 obj (3 rel) or 2 obj (1 rel)
        # task_ids : BS * 6 of each task_id of each example
        return dict(
            mask_img=mask_img_embeds_pt,
            mask_img_augs=mask_img_aug_embeds_pt,
            gt_edge=torch.stack(gt_edge), 
            edge_mask=edge_mask_pt, 
            task_ids=torch.LongTensor(task_ids),
            ex_ids=torch.LongTensor(example_ids),
            shuf_rel_idx=torch.stack(shuffle_rel_idx),
            mask_img_neg=mask_img_embeds_rand,
            edge_mask_neg=edge_mask_rand_pt
            )
    return mask_collate_fn


def get_dataloader(args):
    dataset_train, dataset_val, train_fname, val_fname = get_task_dataset(args)
    dataset_train, task_ids, seen_graphs, taskId2RelId, relName2Id, total_tr_examples, ds_stats = aggregate_tasks(dataset_train, train_fname, max_count=args.max_count)
    
    for k, v in ds_stats.items():
        print(f"{k[0]} objs, {k[1]} dist: {v} examples")
    
    if args.padding_objs == -1:
        args.padding_objs = args.num_objs + args.max_n_distractors
    args.total_tasks = len(taskId2RelId)
    args.total_tr_examples = total_tr_examples
    args.num_rels = len(relName2Id) - 1 #exclude None relation
    aggregate_tasks(dataset_val, val_fname, seen_graphs=seen_graphs)
    args.workers = min(args.workers, os.cpu_count())
    
    if args.max_count and not(total_tr_examples == len(dataset_train) * 6 and total_tr_examples == args.total_tasks * args.max_count * 6):
        print(f"total_tr_examples: {total_tr_examples}, len(dataset_train) * 6: {len(dataset_train) * 6}, args.total_tasks * args.max_count * 6: {args.total_tasks * args.max_count * 6}")
        if args.assert_max_count:
            assert False, "Equality condition failed, likely means at least one task does not have up to {max_count} samples"

    print(task_ids, len(dataset_train))
    print(relName2Id)
    
    train_collate = create_collate_fn(relName2Id, args.num_augs, args.padding_objs, args.train_aug, args.train_mask_shuffle)
    dataloader_train = DataLoader(dataset_train, batch_size=args.task_batch_size, num_workers=args.workers, collate_fn=train_collate, pin_memory=True, shuffle=args.train_data_shuffle, drop_last=args.train_drop_last) #If using whole dataset and not droplast, causes NaN
    
    val_collate = create_collate_fn(relName2Id, 1, args.padding_objs, augmentation=False, shuffle_masks=False)
    dataloader_val = DataLoader(dataset_val, batch_size=args.task_batch_size, num_workers=args.workers, collate_fn=val_collate, pin_memory=True, shuffle=False, drop_last=False)

    max_bs_tr = len(dataset_train) if args.evaluate or args.rel_dim > 3 else args.task_batch_size #Max, HARDCODE
    max_bs_val = len(dataset_val) if args.evaluate or args.rel_dim > 3 else args.task_batch_size #Max, HARDCODE
    dataloader_train_edge = DataLoader(dataset_train, batch_size=max_bs_tr, num_workers=args.workers, collate_fn=val_collate, pin_memory=True, shuffle=False, drop_last=False)
    dataloader_val_edge = DataLoader(dataset_val, batch_size=max_bs_val, num_workers=args.workers, collate_fn=val_collate, pin_memory=True, shuffle=False, drop_last=False)
    
    return dataloader_train, dataloader_train_edge, dataloader_val, dataloader_val_edge, taskId2RelId


def get_task_dataset(args):
    # BabyARC-relation dataset:
    obj3_str = "+3ai+3a+3b" if args.num_objs >= 3 else ""
    obj4_str = "+4a+4ai+4b" if args.num_objs >= 4 else ""
    dataset_args = init_args({
        "dataset": f"h-r^2ai+2a{obj3_str}{obj4_str}:SameShape+SameColor(Line+Rect+RectSolid+Lshape)-d^1:Line+Rect+RectSolid+Lshape",
        "seed": 2,
        "n_examples": args.n_examples,
        "canvas_size": args.canvas_size,
        "rainbow_prob": 0.,
        "color_avail": "-1",
        "max_n_distractors": args.max_n_distractors,
        "min_n_distractors": args.min_n_distractors,
        "allow_connect": True
    })
    dataset_train, dataset_args_post = get_dataset(dataset_args, is_load=True)
    train_fname = os.path.basename(dataset_args.dataset_filename)
    
    dataset_args.n_examples = 1200
    dataset_args.seed = 1
    dataset_val, dataset_args_post2 = get_dataset(dataset_args, is_load=True)
    val_fname = os.path.basename(dataset_args.dataset_filename)

    # Hacky code to patch previously generated datasets
    dataset_train.min_n_distractors = args.min_n_distractors
    dataset_train.max_n_distractors = args.max_n_distractors
    dataset_val.min_n_distractors = args.min_n_distractors
    dataset_val.max_n_distractors = args.max_n_distractors

    return dataset_train, dataset_val, train_fname, val_fname