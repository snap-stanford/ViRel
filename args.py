import os
from easydict import EasyDict as edict
import argparse
import yaml
import ipdb

def get_default_args():
    args = edict()

    args.base_log = "/dfs/user/dzeng/virel_results"
    args.log_dir = None # Generates from args if value is None
    args.pkl_name = "data_rec.p"
    args.force_py_backup = False
    args.ddp = False # Distributed Data Parallel, current broken
    args.random_seed = 0
    
    args.canvas_size = 16
    args.num_objs = 3
    args.min_n_distractors = 0
    args.max_n_distractors = 0
    args.padding_objs = -1

    args.task_batch_size = 168
    args.workers = 25
    args.num_augs = 2
    args.train_aug = True
    args.train_mask_shuffle = True
    args.train_data_shuffle = True
    args.train_drop_last = True

    args.learning_rate = 1e-3
    args.lr_scheduler_type = "None"
    args.momentum = 0.9
    args.weight_decay = 0 #5e-4
    args.num_epochs = 10000

    args.mlp_concat = True
    args.multi_head_rel = False
    args.supervised = False
    args.softmax_rel = False
    args.obj_dim = 100
    args.rel_dim = 20
    
    args.print_interval = 20
    args.print_interval2 = 10 # Frequency to print edge dist and acc
    args.print_interval3 = 25
    args.save_interval = 25 # Frequency to save model
    args.write_tb_imgs = False
    
    args.intra_weight = 0 
    args.inter_weight = 0 
    args.edge_rep_weight = 0
    args.edge_att_weight = 0
    
    args.classify = False
    args.cl_weight = 1

    args.freeze_gin = False
    args.gin_last_act = False

    args.is_lookup_mask = False
    args.alpha_entr = 0

    args.is_ixz = False
    args.ixz_gin_mean = False
    args.reparam_mode = "diag"
    args.ixz_weight = 0.1

    args.is_inner_loop = False
    args.inner_lr = 0.5
    args.num_inner_loop = 5

    args.run_id = "empty_id"
    args.force_overwrite = True
    args.load_latest = True
    args.checkpoint_step = -1
    args.evaluate = False
    args.fine_tune = False
    args.show_tsne = True

    # Autofilled in args
    args.num_rels = None
    args.total_tasks = None
    args.total_tr_examples = None

    return args

def open_yaml(args, config_file):
    args = edict(args)
    with open(config_file, "rb") as f:
        yaml_config = yaml.safe_load(f)
        args.update(yaml_config)
    return args

def write_yaml(args, config_file):
    with open(config_file, "w") as f:
        # print(os.path.abspath(config_file))
        # print(f"Written {config_file}")
        yaml.dump(args, f)

def validate_args(args, warn_def_args=True):
    def_args = get_default_args()
    extra_args = args.keys() - def_args.keys()
    extra_def_args = def_args.keys() - args.keys()
    
    if extra_args:
        print(f"Extra keys in args: {extra_args}")
    if warn_def_args and extra_def_args:
        print(f"Keys not specified in args: {extra_def_args}")

def update_with_default(args):
    def_args = get_default_args()
    for k in (def_args.keys() - args.keys()):
        args[k] = def_args[k]
    return args


def standardize_args(args):
    if not args.is_lookup_mask:
        args.alpha_entr = 0

    if not args.is_ixz:
        args.ixz_gin_mean = False
        args.reparam_mode = ""
        args.ixz_weight = 0

    if not args.is_inner_loop:
        args.inner_lr = 0
        args.num_inner_loop = 0

    if args.classify:
        if args.intra_weight == 0 and args.inter_weight == 0:
            print("Changing intra_weight and inter_weight to 0 due to classify flag")
        args.intra_weight = 0
        args.inter_weight = 0
    else:
        args.cl_weight = 0

    if not args.max_count:
        args.assert_max_count = False

    return args

def parse_args():
    """Parse CLI arguments.
    https://sungwookyoo.github.io/tips/ArgParser/
    """
    parser = argparse.ArgumentParser(description="Relation Analogy with GNN Isomorphism")
    parser.add_argument('--yaml', '-y', help="yaml configuration file *.yml", type=str)
    # parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
    
    def_args = get_default_args()
    for arg_name, val in def_args.items():
        arg_name = arg_name.replace("_", "-")
        parser.add_argument(f"--{arg_name}", type=type(val), default=val, help=arg_name)
    
    args = parser.parse_args()

    # Precedence: args object, then yaml file, then CLI args
    if args.yaml:
        print(f"Loading args from yaml: {args.yaml}")
        args = open_yaml(def_args, args.yaml)
    else:
        print("Loading args from CLI")
    
    args.local_rank = int(os.getenv("LOCAL_RANK", -1))
    args = standardize_args(args)

    return args

def get_args(args):
    if args is None:
        args = parse_args()
    else:
        validate_args(args)
        update_with_default(args)
        standardize_args(args)
    return args
