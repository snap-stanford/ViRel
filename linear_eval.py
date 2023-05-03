"""
Script for training EBMs for discovering concepts, relations and operators.
"""
from copy import deepcopy
import itertools
import logging

import os
import sys
import pdb
import ipdb
import pickle
import pprint as pp
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
logging.getLogger('matplotlib.font_manager').disabled = True
import time
from collections import defaultdict as ddict
from functools import lru_cache
import numpy as np

import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '../../..'))

from models.task_model import get_gnn_model
from data import get_dataloader
from loss import graph_task_loss
from args import get_args
from kmeans import kmeans_with_initial_prototypes, kmeans, find_nearest_prototypes
from util import *

def fine_tune_main(train_loader_edge, val_loader_edge, model, log_dir, args, device):
    if args.load_embeddings:
        # Extract edge embeddings and save to disk
        # Alterantively 
        train_edges = load(log_dir, None)
        val_edges = load(log_dir, None)
    else:
        train_edges = extract_embeddings(train_loader_edge, model, args, device)
        val_edges = extract_embeddings(val_loader_edge, model, args, device)

        # Save the edges somewhere
        save(train_edges)
        save(val_edges)
    

    linear = nn.Linear(args.num_rels, args.total_tasks)
    linear, device = model_parallel(linear, args)

    optimizer = optim.Adam(model.parameters(), 
        lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(reduction='none') # Classification loss function
    
    # Train linear layer on embeddings
    best_acc = 0
    start_step = 0
    start_epoch = 0
    writer = None
    data_record = None
    for epoch in range(start_epoch, args.num_ft_epochs):
        linear_train(train_loader_edge, train_edges, linear, loss_fn, optimizer, epoch, start_step, writer, data_record, args, device)

def extract_embeddings(loader_edge, model, args, device):
    # Loop through dataloader, save forward results into a tensor 
    pass

def linear_train(loader, model, loss_fn, optimizer, epoch, start_step, writer, data_record, args, device):
    
    pass



# Create loader from extracted embeddings