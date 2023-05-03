from collections import OrderedDict, Counter
from copy import deepcopy
import itertools
import json
import matplotlib.pylab as plt
from multiset import Multiset
from numbers import Number
import numpy as np
import pdb
import scipy as sp
from scipy import ndimage
import torch
import torch.nn as nn
import yaml
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from pytorch_net.util import to_Variable_recur, to_np_array, record_data, remove_duplicates, split_string, broadcast_keys, make_dir, filter_filename, TopKList, get_device

color_dict = {0: [1, 1, 1],
              1: [0, 0, 1],
              2: [1, 0, 0],
              3: [0, 1, 0],
              4: [1, 1, 0],
              5: [.5, .5, .5],
              6: [.5, 0, .5],
              7: [1, .64, 0],
              8: [0, 1, 1],
              9: [.64, .16, .16],
              10: [1, 0, 1],
              11: [.5, .5, 0],
             }


# color_dict = {0: [0, 0, 0],
#               1: [0, 0, 0.9],
#               2: [0.9, 0, 0],
#               3: [0, 0.9, 0],
#               4: [0.9, 0.9, 0],
#               5: [.5, .5, .5],
#               6: [.5, 0, .5],
#               7: [0.9, .64, 0],
#               8: [0, 0.9, 0.9],
#               9: [0.9, 0, 0.9],
#              }


def onehot_to_RGB(tensor, scale=1):
    """Transform 10-channel ARC image to 3-channel RGB image.

    Args:
        tensor: [B, C:10, H, W]

    Returns:
        tensor: [B, C:3, H, W]
    """
    tensor = torch.LongTensor(to_np_array(tensor)).argmax(1)  # [B, C:10, H, W] -> [B, H, W]
    collection = torch.FloatTensor(list(color_dict.values())) * scale  # [10, 3]
    return collection[tensor].permute(0,3,1,2)  # collection[tensor]: [B, H, W, C:3]; after permute: [B, C:3, H, W]


def to_one_hot(tensor, n_channels=10):
    """
    Args:
        tensor: [[B], H, W], where each values are integers in 0,1,2,... n_channels-1

    Returns:
        tensor: [[B], n_channels, H, W]
    """
    if isinstance(tensor, torch.Tensor):
        collection = torch.eye(n_channels)   # [n_channels, n_channels]
        if len(tensor.shape) == 2:  # [H, W]
            return collection[tensor.long()].permute(2,0,1)  # [C, H, W]
        elif len(tensor.shape) == 3: # [B, H, W]
            return collection[tensor.long()].permute(0,3,1,2)  # [B, C, H, W]
        else:
            raise Exception("tensor must be 2D or 3D!")
    elif isinstance(tensor, np.ndarray):
        collection = np.eye(n_channels)
        if len(tensor.shape) == 2:  # [H, W]
            return collection[tensor.astype(int)].transpose(2,0,1)
        elif len(tensor.shape) == 3:  # [H, W]
            return collection[tensor.astype(int)].transpose(0,3,1,2)
        else:
            raise Exception("tensor must be 2D or 3D!")
    else:
        raise Exception("tensor must be PyTorch or numpy tensor!")


def visualize_masks(imgs, masks, recons, vis):
    # print('recons min/max', recons[:, 0].min().item(), recons[:, 0].max().item())
    # print('recons1 min/max', recons[:, 1].min().item(), recons[:, 1].max().item())
    # print('recons2 min/max', recons[:, 2].min().item(), recons[:, 2].max().item())
    if imgs.shape[1] != 3 and recons.shape[1] != 3:
        imgs = onehot_to_RGB(imgs, scale=1)
        recons = onehot_to_RGB(recons, scale=1)

    recons = np.clip(recons, 0., 1.)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    colors.extend([(c[0]//2, c[1]//2, c[2]//2) for c in colors])
    colors.extend([(c[0]//4, c[1]//4, c[2]//4) for c in colors])
    seg_maps = np.zeros_like(imgs)
    masks = np.argmax(masks, 1)
    for i in range(imgs.shape[0]):
        for y in range(imgs.shape[2]):
            for x in range(imgs.shape[3]):
                seg_maps[i, :, y, x] = colors[masks[i, y, x]]

    seg_maps /= 255.0
    vis.images(np.concatenate((imgs, seg_maps, recons), 0), nrow=imgs.shape[0])


def to_tuple(tensor):
    """Transform a PyTorch tensor into a tuple of ints."""
    assert len(tensor.shape) == 1
    return tuple([int(item) for item in tensor])


def get_root_dir():
    dirname = os.getcwd()
    dirname_split = dirname.split("/")
    index = dirname_split.index("reasoning")
    dirname = "/".join(dirname_split[:index + 1])
    return dirname


def load_dataset(filename, directory=None, isTorch=True):
    if directory is None:
        directory = 'ARC/data/training'
    with open(os.path.join(get_root_dir(), directory, filename)) as json_file:
        dataset = json.load(json_file)
    if isTorch:
        dataset = to_Variable_recur(dataset, type="long")
    return dataset

def get_filenames(path, include=""):
    if not isinstance(include, list):
        include = [include]
    with os.scandir(path) as it:
        filenames = []
        for entry in it:
            if entry.is_file():
                is_in = True
                for element in include:
                    if element not in entry.name:
                        is_in = False
                        break
                if is_in:
                    filenames.append(entry.name)
        return filenames
    
def get_directory_tasks(directories, split_json=False, include_dir=False):
    file_list = []
    for directory in directories:
        files = get_filenames(os.path.join(get_root_dir(), directory),  include=".json")
        file_list += [(directory, file.split('.json')[0] if split_json else file) for file in files]
    if not include_dir:
        file_list = [file for _, file in file_list]
        # Important: Sort the list of files to make sure the order of files is consistent
        file_list.sort()
    return file_list

def sample_dataset(directories, task_list=None):
    """Sample one dataset from the given directories.
    """
    file_list = []
    for directory in directories:
        files = get_filenames(os.path.join(get_root_dir(), directory), include=".json")
        file_list += [(directory, file) for file in files]
    # Check if the file is in task_list:
    if task_list is not None:
        file_list_union = []
        for directory, file in file_list:
            if file.split(".json")[0] in task_list:
                file_list_union.append((directory, file))
    else:
        file_list_union = file_list
    assert len(file_list_union) > 0, "Did not find task {} in {}.".format(task_list, directories)
    id_chosen = np.random.choice(len(file_list_union))
    directory_chosen, file_chosen = file_list_union[id_chosen]
    dataset = load_dataset(file_chosen, directory=directory_chosen)
    return dataset, file_chosen


def get_task_list(task_file):
    """Obtain task_list from task_file."""
    task_list = []
    with open(os.path.join(get_root_dir(), task_file), "r") as f:
        for line in f.readlines():
            if len(line) > 1:
                line_core = line.split("#")[0].strip()
                if len(line_core) > 1:
                    task_list.append(line.split("#")[0].strip())
    return task_list


def get_inputs_targets(dataset):
    """Get inputs (list of OrderedDict) and targets (OrderedDict) from train or test dataset."""
    inputs = OrderedDict()
    targets = OrderedDict()
    for i in range(len(dataset)):
        input, target = dataset[i]["input"], dataset[i]["output"]
        inputs[i] = input
        targets[i] = target
    inputs = [inputs]
    return inputs, targets


def get_inputs_targets_EBM(dataset):
    """Get inputs (list of OrderedDict) and targets (OrderedDict) from ConceptCompositionDataset."""
    inputs = OrderedDict()
    targets = OrderedDict()
    infos = OrderedDict()
    for i in range(len(dataset)):
        if isinstance(dataset[i][0], tuple) or isinstance(dataset[i][0], list):
            assert len(dataset[i][0]) == 2
            input = dataset[i][0][0]
            target = dataset[i][0][1]
        else:
            input = target = dataset[i][0]
        inputs[i] = input
        targets[i] = target
        infos[i] = dataset[i][3]
    inputs = [inputs]
    return inputs, targets, infos


def plot_with_boundary(image, plt):
    im = plt.imshow(image, interpolation='none', vmin=0, vmax=1, aspect='equal');
#     height, width = np.array(image).shape[:2]
#     ax = plt.gca();

#     # Major ticks
#     ax.set_xticks(np.arange(0, width, 1));
#     ax.set_yticks(np.arange(0, height, 1));

#     # Labels for major ticks
#     ax.set_xticklabels(np.arange(1, width + 1, 1));
#     ax.set_yticklabels(np.arange(1, height + 1, 1));

#     # Minor ticks
#     ax.set_xticks(np.arange(-.5, width, 1), minor=True);
#     ax.set_yticks(np.arange(-.5, height, 1), minor=True);

#     # Gridlines based on minor ticks
#     ax.grid(which='minor', color='w', linestyle='-', linewidth=2)


def visualize_matrices(matrices, num_rows=None, row=0, images_per_row=None, plt=None, is_show=True, filename=None, title=None, subtitles=None, use_color_dict=True, masks=None, is_round_mask=False, **kwargs):
    """
    :param matrices: if use_color_dict is False, shape [N, 3, H, W] where each value is in [0, 1]
                     otherwise, shape [N, H, W] where each value is a number from 0-10
                     if masks is not None, shape [N, 1, H, W] where each value is in [0, 1]
    """
    if images_per_row is None:
        images_per_row = min(6, len(matrices))
    num_plots = len(matrices)
    if num_rows is None:
        num_rows = int(np.ceil(num_plots / images_per_row))
    if plt is None:
        import matplotlib.pylab as plt
        plt_show = True
        plt.figure(figsize=(3*images_per_row, 3*num_rows))
    else:
        plt_show = False

    for i, matrix in enumerate(matrices):
        matrix = to_np_array(matrix, full_reduce=False)
        if isinstance(matrix, bool):
            continue
        if not use_color_dict:
            plt.subplot(num_rows, images_per_row, i + 1 + 2 * row)
            image = np.zeros((matrix.shape[-2], matrix.shape[-1], 3))
            for k in range(matrix.shape[-2]):
                for l in range(matrix.shape[-1]):
                    image[k, l] = np.array(matrix[:, k, l])
        else:
            if matrix.dtype.name.startswith("float"):
                # Typically for masks, we want to round them such that a 0.5+ value
                # is considered as being "part of" the mask.
                matrix = np.round(matrix).astype("int")
            if matrix.dtype.name.startswith("int") or matrix.dtype.name == "bool":
                plt.subplot(num_rows, images_per_row, i + 1 + 2 * row)
                image = np.zeros((*matrix.shape, 3))
                for k in range(matrix.shape[0]):
                    for l in range(matrix.shape[1]):
                        image[k, l] = np.array(color_dict[matrix[k, l]])

        if masks is not None:
            mask = to_np_array(masks[i], full_reduce=False)
            for k in range(mask.shape[-2]):
                for l in range(mask.shape[-1]):
                    image[k, l] *= (np.around(mask[:, k, l]) if is_round_mask else mask[:, k, l])
        plot_with_boundary(image, plt)
        if subtitles is not None:
            plt.title(subtitles[i])
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            right=False,
            labelsize=0,
            labelbottom=False)
        # plt.axis('off')
    ax = plt.gca()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    if title is not None:
        plt.suptitle(title, fontsize=14, y=0.9)
    if filename is not None:
        ax = plt.gca()
        ax.set_rasterized(True)
        plt.savefig(filename, bbox_inches="tight", dpi=400, **kwargs)
    if is_show and plt_show:
        plt.show()


def visualize_dataset(dataset, filename=None, is_show=True, title=None, **kwargs):
    def to_value(input):
        if not isinstance(input, torch.Tensor):
            input = input.get_node_value()
        return input
    length = len(dataset["train"]) + 1
    plt.figure(figsize=(7, 3.5 * (length)))
    for i, data in enumerate(dataset["train"]):
        visualize_matrices([to_value(data["input"]), to_value(data["output"])], images_per_row=2, num_rows=length, row=i, plt=plt, is_show=is_show, title=title if i == 0 else None)
    if "test" in dataset:
        visualize_matrices([to_value(dataset["test"][0]["input"])], images_per_row=2, num_rows=length, row=i+1, plt=plt, is_show=is_show)
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", **kwargs)
    if is_show:
        plt.show()
        
        
def plot_matrices(matrices):
    if not isinstance(matrices, list):
        matrices = [matrices]
    if len(matrices[0].shape) == 2:
        visualize_matrices(matrices)
    elif len(matrices[0].shape) == 3:
        length = len(matrices)
        plt.figure(figsize=(3.5*length, 4))
        for i, matrix in enumerate(matrices):
            vmax = 255 if matrix.max() > 1 else 1
            plt.subplot(1, length, 1 + i)
            plt.imshow(to_np_array(matrix).transpose(1, 2, 0).astype(int), vmax=vmax)
            plt.axis('off')
        plt.show()


def get_op_shape(result):
    """Get the shape of the result on an operator."""
    device = result[list(result.keys())[0]].device
    op_shape_dict = OrderedDict()
    for key, item in result.items():
        op_shape = torch.zeros(2).long().to(device)
        if not (isinstance(item, torch.Tensor) or isinstance(item, np.ndarray)):
            item = item.get_node_value()
        if hasattr(item, "shape"):
            shape = item.shape
            if len(shape) == 2:
                op_shape = torch.LongTensor(tuple(item.shape)).to(device)
            elif len(shape) == 1:
                op_shape[0] = torch.LongTensor(tuple(item.shape))
        op_shape_dict[key] = op_shape
    return op_shape_dict


def combine_pos(*pos_list):
    """Obtain a minimum bounding box in the form of 'pos' to the list of input pos."""
    pos_list = np.array([to_np_array(pos) for pos in pos_list])
    pos_list[:, 2] += pos_list[:, 0]
    pos_list[:, 3] += pos_list[:, 1]

    new_pos_min = pos_list[:, :2].min(0)
    new_pos_max = pos_list[:, 2:].max(0)
    new_pos = tuple(np.concatenate([new_pos_min, new_pos_max - new_pos_min]).round().astype(int).tolist())
    return new_pos


def to_Graph(dataset, base_concept):
    dataset_Graph = deepcopy(dataset)
    for mode in ["train", "test"]:
        for i in range(len(dataset[mode])):
            for key in ["input", "output"]:
                dataset_Graph[mode][i][key] = base_concept.copy().set_node_value(dataset[mode][i][key])
                if base_concept.name == "Image":
                    dataset_Graph[mode][i][key].set_node_value([0, 0, dataset[mode][i][key].shape[0], dataset[mode][i][key].shape[1]], "pos")
    return dataset_Graph


def masked_equal(input1, input2, exclude=0):
    """Return True for each elements only if the corresponding elements are equal and nonzero."""
    if exclude is None:
        return input1 == input2
    else:
        nonzero_mask = (input1 != exclude) | (input2 != exclude)
        return (input1 == input2) & nonzero_mask


def get_input_output_mode_dict(operators, is_inherit=True):
    """Get the dictionary of input (output) modes mapped to the body operator."""
    concepts = combine_dicts(operators)
    input_mode_dict = {}
    output_mode_dict = {}
    for mode, operator in operators.items():
        if operator.__class__.__name__ == "Graph":
            # Input dict:
            input_modes = tuple(sorted([input_node.split(":")[-1] for input_node in operator.input_nodes.keys()]))
            record_data(input_mode_dict, [mode], [input_modes])

            # Output dict:
            output_mode = operator.get_output_nodes(types=["fun-out"], allow_goal_node=True)[0].split(":")[-1]
            record_data(output_mode_dict, mode, output_mode, ignore_duplicate=True)
    return input_mode_dict, output_mode_dict


def get_inherit_modes(mode, concepts, type=None):
    """Get all the modes that is inherit from (or to) the current mode."""
    mode = split_string(mode)[0]
    modes_inherit = [mode]
    if type == "from":
        if hasattr(concepts[mode], "inherit_from"):
            modes_inherit += concepts[mode].inherit_from
    elif type == "to":
        if hasattr(concepts[mode], "inherit_from"):
            modes_inherit += concepts[mode].inherit_from
    else:
        raise
    return modes_inherit


def combine_dicts(dicts):
    """Combine multiple concepts into a single one."""
    if isinstance(dicts, list):
        dicts_cumu = {}
        for dicts_ele in dicts:
            dicts_cumu.update(dicts_ele)
        dicts = dicts_cumu
    return dicts


def accepts(operator_input_modes, input_modes, concepts, mode="exists"):
    """Check if the operator (specified by operator_input_modes) accepts input_modes."""
    # If concepts is a list of concepts dictionaries, accumulate them:
    concepts = combine_dicts(concepts)

    if mode == "fully-cover":
        """True only if the operator's input can fully cover the input_modes"""
        if len(operator_input_modes) < len(input_modes):
            return False
        # Get input_modes_all:
        input_modes_all = []
        for mode in input_modes:
            mode_inherit = get_inherit_modes(mode, concepts, type="from")
            input_modes_all.append(mode_inherit)
        is_accept = False
        for comb_ids in itertools.combinations(range(len(operator_input_modes)), len(input_modes)):
            selected_operator_modes = [operator_input_modes[id] for id in comb_ids]
            is_accept_selected = False
            for keys in itertools.product(*input_modes_all):
                if Multiset(keys).issubset(Multiset(selected_operator_modes)):
                    is_accept_selected = True
                    break
            if is_accept_selected:
                is_accept = True
                break
    elif mode == "exists":
        """is_accept is True as long as there is an input_mode that can fed to the operator."""
        is_accept = False
        for input_mode in input_modes:
            mode_inherit = get_inherit_modes(input_mode, concepts, type="from")
            is_accept_selected = False
            for mode in mode_inherit:
                if mode in operator_input_modes:
                    is_accept_selected = True
                    break
            if is_accept_selected:
                is_accept = True
                break
    else:
        raise
    return is_accept


def find_valid_operators(nodes, operators, concepts, input_mode_dict, arity=2, exclude=None):
    """Given out_nodes, find operators that is compatible to {arity} number of them."""
    assert arity <= len(nodes)
    concepts = combine_dicts(concepts)
    if exclude is None:
        exclude = []
    modes = [node.split(":")[-1] for node in nodes]
    modes_all = []
    valid_options = {}
    for mode in modes:
        modes_inherit = get_inherit_modes(mode, concepts, type="from")
        modes_all.append(modes_inherit)

    for comb_ids in itertools.combinations(range(len(modes)), arity):
        selected_modes = [modes_all[id] for id in comb_ids]
        for keys in itertools.product(*selected_modes):
            for operator_input_modes in input_mode_dict:
                if Multiset(keys).issubset(Multiset(operator_input_modes)):
                    for operator_name in input_mode_dict[operator_input_modes]:
                        if operator_name in valid_options or operator_name in exclude:
                            continue
                        input_nodes = list(operators[operator_name].input_nodes.keys())
                        input_modes = [input_node.split(":")[-1] for input_node in input_nodes]
                        assign_list = []
                        chosen = []
                        """
                        selected_modes: [['Line', 'Image'], ['Line', 'Image']]  # The original mode and all the modes it inherit from, considering n choose k
                        chosen_modes: ["Line", "Line"]                          # The original mode considering n choose k
                        keys: ["Image", "Line"]                                 # The specific combination selected that has instance in INPUT_MODE_DICT
                        input_nodes: ['concept1:Line', 'concept1:Image']        # The input nodes for the specific operator
                        input_modes: ["Line", "Image"]                          # The input modes for the specific operator
                        """
                        chosen_nodes = [nodes[id] for id in comb_ids]
                        for j, node in enumerate(chosen_nodes):
                            chosen_key = keys[j]
                            # Assign input_mode to the chosen_modes:
                            for k, input_mode in enumerate(input_modes):
                                if input_mode == chosen_key and k not in chosen:
                                    assign_list.append([node, input_nodes[k]])
                                    break
                            chosen.append(k)

                        valid_options[operator_name] = assign_list
    return valid_options


def canonical(mode):
    """Return canonical mode."""
    return mode.split("*")[-1]


def view_values(Dict):
    """View the value in each item of Dict."""
    new_dict = OrderedDict()
    for key, item in Dict.items():
        if isinstance(item, Concept):
            item = item.get_root_value()
        new_dict[key] = deepcopy(item)
    return new_dict


def get_last_output(results, is_Graph=True):
    """Obtain the last output as a node in results"""
    output = results[list(results.keys())[-1]]
    if is_Graph and len(output.shape) == 2:
        output = concepts["Image"].copy().set_node_value(output)
    return output


def canonicalize_keys(Dict):
    """Make sure the keys of Dict have the same length. If not, make it so."""
    keys_core = [key if isinstance(key, tuple) else (key,) for key in Dict]
    length_list = [len(key_core) for key_core in keys_core]

    assert len(np.unique(length_list)) <= 2
    if len(np.unique(length_list)) <= 1:
        return Dict
    else:
        idx_max = max(np.unique(length_list))
        idx_min = min(np.unique(length_list))
        assert idx_max - idx_min == 1
        str_list = []
        for key_core in keys_core:
            if len(key_core) == idx_max:
                str_list.append(key_core[-1].split("-")[0])
        assert len(np.unique(str_list)) == 1
        string = "{}-{}".format(str_list[0], 0)
        key_map = {}
        for key, key_core in zip(list(Dict.keys()), keys_core):
            if len(key_core) == idx_min:
                new_key = key_core + (string,)
                key_map[key] = new_key

        for key, new_key in key_map.items():
            Dict[new_key] = Dict.pop(key)
    return Dict


def get_first_key(key):
    if isinstance(key, tuple):
        return key[0]
    else:
        assert isinstance(key, Number)
        return key


def get_Dict_with_first_key(Dict):
    if isinstance(Dict, OrderedDict):
        return OrderedDict([[get_first_key(key), item] for key, item in Dict.items()])
    else:
        return {get_first_key(key): item for key, item in Dict.items()}


def broadcast_inputs(inputs):
    """Broadcast inputs."""
    is_dict = False
    for input_arg in inputs:
        if isinstance(input_arg, dict):
            is_dict = True
            break
    if not is_dict:
        inputs = [{0: input_arg} for input_arg in inputs]

    # Broadcast input keys, and record inputs into results:
    input_key_list_all = []
    for i, input_arg in enumerate(inputs):
        if isinstance(input_arg, dict):
            input_key_list_all.append(input_arg.keys())
        else:
            input_key_list_all.append(None)
    input_key_dict = broadcast_keys(input_key_list_all)
    input_keys = list(input_key_dict.keys())
    return inputs, input_keys


def get_attr_proper_name(concept, node_name):
    length = node_name.find(concept.name.lower())
    if length == -1:
        return "^".join(node_name.split("^")[1:])
    else:
        return "^".join(node_name[length:].split("^")[1:])


def get_run_status():
    """Get the runing status saved in /web/static/states/run_status.txt, with 0 meaning to stop, and other string meaning the method being run."""
    run_status = "0"
    status_path = get_root_dir() + "/web/static/states/run_status.txt"
    make_dir(status_path)
    if not os.path.exists(status_path):
        with open(status_path, "w") as f:
            f.write("0")
    with open(status_path, "r") as f:
        run_status = f.readline()
    return run_status


def set_run_status(status):
    status_path = get_root_dir() + "/web/static/states/run_status.txt"
    make_dir(status_path)
    if not os.path.exists(status_path):
        with open(status_path, "w") as f:
            f.write("")
    with open(status_path, "w") as f:
        f.write("{}".format(status))


class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.exp()


class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.square()


def get_activation(activation, inplace=False):
    if activation.lower() == "relu":
        return nn.ReLU(inplace=inplace)
    elif activation.lower() == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=inplace)
    elif activation.lower() == "leakyrelu0.2":
        return nn.LeakyReLU(negative_slope=0.2, inplace=inplace)
    elif activation.lower() == "swish":
        return nn.SiLU(inplace=inplace)
    elif activation.lower() == "tanh":
        return nn.Tanh(inplace=inplace)
    elif activation.lower() == "sigmoid":
        return nn.Sigmoid(inplace=inplace)
    elif activation.lower() == "linear":
        return nn.Identity()
    elif activation.lower() == "elu":
        return nn.ELU(inplace=inplace)
    elif activation.lower() == "softplus":
        return nn.Softplus()
    elif activation.lower() == "rational":
        return Rational()
    elif activation.lower() == "exp":
        return Exp()
    elif activation.lower() == "square":
        return Square()
    else:
        raise


def get_normalization(normalization_type, in_channels=None):
    if normalization_type.lower() == "none":
        return nn.Identity()
    elif normalization_type.lower().startswith("gn"):
        n_groups = eval(normalization_type.split("-")[1])
        return nn.GroupNorm(n_groups, in_channels, affine=True)
    elif normalization_type.lower() == "in":
        return nn.InstanceNorm2d(in_channels, affine=True)
    else:
        raise


class Rational(torch.nn.Module):
    """Rational Activation function.
    Implementation provided by Mario Casado (https://github.com/Lezcano)
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are initialized to the best rational 
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """
    def __init__(self):
        super().__init__()
        self.coeffs = torch.nn.Parameter(torch.Tensor(4, 2))
        self.reset_parameters()

    def reset_parameters(self):
        self.coeffs.data = torch.Tensor([[1.1915, 0.0],
                                         [1.5957, 2.383],
                                         [0.5, 0.0],
                                         [0.0218, 1.0]])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0,1].zero_()
        exp = torch.tensor([3., 2., 1., 0.], device=input.device, dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        output = torch.div(PQ[..., 0], PQ[..., 1])
        return output


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        n_neurons,
        n_layers,
        activation="relu",
        output_size=None,
        last_layer_linear=True,
    ):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.activation = activation
        if n_layers > 1 or last_layer_linear is False:
            self.act_fun = get_activation(activation)
        self.output_size = output_size if output_size is not None else n_neurons
        self.last_layer_linear = last_layer_linear
        for i in range(1, self.n_layers + 1):
            setattr(self, "layer_{}".format(i), nn.Linear(
                self.input_size if i == 1 else self.n_neurons,
                self.output_size if i == self.n_layers else self.n_neurons,
            ))
            torch.nn.init.xavier_normal_(getattr(self, "layer_{}".format(i)).weight)

    def forward(self, x):
        for i in range(1, self.n_layers + 1):
            x = getattr(self, "layer_{}".format(i))(x)
            if i != self.n_layers or not self.last_layer_linear:
                x = self.act_fun(x)
        return x


def to_device_recur(iterable, device, is_detach=False):
    if isinstance(iterable, list):
        return [to_device_recur(item, device, is_detach=is_detach) for item in iterable]
    elif isinstance(iterable, tuple):
        return tuple(to_device_recur(item, device, is_detach=is_detach) for item in iterable)
    elif isinstance(iterable, dict):
        return {key: to_device_recur(item, device, is_detach=is_detach) for key, item in iterable.items()}
    elif hasattr(iterable, "to"):
        iterable = iterable.to(device)
        if is_detach:
            iterable = iterable.detach()
        return iterable
    else:
        if hasattr(iterable, "detach"):
            iterable = iterable.detach()
        return iterable


def select_item(list_of_lists, indices=None):
    """Choose items from list of lists using indices, assuming that each element list has the same length."""
    flattened_list = []
    if indices is None:
        for List in list_of_lists:
            flattened_list += List
    else:
        for index in indices:
            q, r = divmod(index, len(list_of_lists[0]))
            flattened_list.append(list_of_lists[q][r])
    return flattened_list


def action_equal(action, num, allowed_modes):
    """Return True if action == num and action is in allowed modes.
    allowed_modes: string, e.g. "012".
    """
    action_int = int(action)
    return str(action_int) in allowed_modes and action_int == num


################################
# For parsing Rect and Lines:
################################

def get_empty_neighbor(matrix, i, j):
    """Calculate the number of empty neighbors of the current pixel.
    If the pixel has value, return which direction of the pixel has value
    If the pixel is 0, return 0"""
    m = len(matrix)
    n = len(matrix[0])
    empty_neighbor = 0
    if matrix[i][j]:
        up, down, left, right = True, True, True, True
        if i == 0 or matrix[i - 1][j] == 0:
            empty_neighbor += 1
            up = False
        if i == m - 1 or matrix[i + 1][j] == 0:
            empty_neighbor += 1
            down = False
        if j == 0 or matrix[i][j - 1] == 0:
            empty_neighbor += 1
            left = False
        if j == n - 1 or matrix[i][j + 1] == 0:
            empty_neighbor += 1
            right = False
        return empty_neighbor, up, down, left, right

    # Return 0 if the pixel has no value
    return 0, False, False, False, False


def add_dict(original_list, new_ele):
    """Add a new element to the dictionary according to its shape. 
    Return the updated list."""
    if new_ele[2] == 1 and new_ele[3] == 1:
        key = 'Pixel'
    elif new_ele[2] == 1 or new_ele[3] == 1:
        key = 'Line'
    else:
        key = 'RectSolid'
    if key not in original_list:
        original_list[key] = []
    original_list[key].append(new_ele)
    return original_list


def eliminate_rectangle(matrix, pos):
    """Eliminate an rectangle from a matrix.
    Return the updated matrix."""
    new_matrix = np.zeros_like(matrix)
    for i in range(pos[2]):
        for j in range(pos[3]):
            new_matrix[pos[0] + i][pos[1] + j] = 1
    return matrix - new_matrix


def maximal_rectangle(matrix, result_list, is_rectangle):
    """Find the rectangle, line, or pixel with the maximal area using dynamic programming.
    Add the maximal rectangle to the dictionary.
    Return the matrix that eliminate this rectangle, the updated dictionary, and whether an object is found."""
    m = len(matrix)
    n = len(matrix[0])

    left = [0] * n
    right = [n] * n
    height = [0] * n

    maxarea = 0
    result = (0, 0, 0, 0)

    for i in range(m):
        cur_left, cur_right = 0, n
        # update height
        for j in range(n):
            if matrix[i][j] == 0:
                height[j] = 0
            else:
                height[j] += 1
        # update left
        for j in range(n):
            if matrix[i][j] == 0:
                left[j] = 0
                cur_left = j + 1
            else:
                left[j] = max(left[j], cur_left)
        # update right
        for j in range(n-1, -1, -1):
            if matrix[i][j] == 0:
                right[j] = n
                cur_right = j
            else:
                right[j] = min(right[j], cur_right)
        # update the area
        for j in range(n):
            if is_rectangle and (height[j] < 2 or (right[j] - left[j]) < 2):
                continue
            tmp = height[j] * (right[j] - left[j])
            if tmp > maxarea:
                maxarea = tmp
                result = (i - height[j] + 1, left[j], height[j], right[j] - left[j])
    # define a matrix with only the max rectangle region has value
    new_matrix = matrix.copy()
    is_found = False
    if result[2] and result[3]:
        new_matrix = eliminate_rectangle(matrix, result)
        result_list = add_dict(result_list, result)
        is_found = True
    return new_matrix, result_list, is_found


def seperate_concept(matrix):
    """Seperate the rectangles, lines, and pixels in a matrix while prioritizing rectangles.
    Return a dictionary of infomations of rectangles, lines, and pixels."""
    matrix = to_np_array(matrix)
    new_matrix = matrix.copy()
    m = len(matrix)
    n = len(matrix[0])
    result = {}
    # Find the maximal rectangle until no rectangles are left in the matrix.
    is_found = True
    while is_found:
        new_matrix, result, is_found = maximal_rectangle(new_matrix, result, True)
    # Find all other lines and pixels left in the matrix.
    is_found = True
    while is_found:
        new_matrix, result, is_found = maximal_rectangle(new_matrix, result, False)
    # Make sure that there are no duplicates:
    for key in ["Pixel", "Line", "RectSolid"]:
        if key in result:
            result[key] = list(set(result[key]))
    return result


################################


def compose_dir_task_source(T_item, show_warning=False):
    if isinstance(T_item, list):
        directories = ["ARC/data/training", "ARC/data/evaluation"]
        task_source = T_item
    elif isinstance(T_item, set):
        directories = list(T_item)
        task_source = None
    elif isinstance(T_item, str):
        directories = ["ARC/data/training", "ARC/data/evaluation"]
        task_source = T_item
    else:
        raise
    # Verify the task directory hash to make sure no tasks have been 
    # added or removed
    for directory in directories:
        file_list = get_directory_tasks([directory])
        hash_str = get_hashing(str(file_list))
        directory = directory + '/' if directory[-1] != '/' else directory
        filename = directory + 'hash.txt'
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                prev_hash = f.readlines()[0].strip()
                assert prev_hash == hash_str, "Hash string doesn't match with saved hash!"
        else:
            try:
                with open(filename, 'w') as f:
                    print(hash_str, file=f)
            except:
                if show_warning:
                    print('Warning: Could not write hash to directory: {}'.format(directory))
                pass
    return directories, task_source


def get_patch(tensor, pos):
    """Get a patch of the tensor based on pos."""
    return tensor[..., int(pos[0]): int(pos[0] + pos[2]), int(pos[1]): int(pos[1] + pos[3])]


def set_patch(tensor, patch, pos, value=None):
    """If value is None, set the certain parts of the tensor with the **non-background** part of the patch.
    Otherwise, set the certain parts of the tensor at the value given according to
    the position of the patch and the **non-background** part.
    """
    pos = [int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])]
    shape = tensor.shape
    pos_0_offset = - pos[0] if pos[0] < 0 else 0
    pos_02_offset = shape[-2] - (pos[0] + pos[2]) if pos[0] + pos[2] > shape[-2] else 0
    pos_1_offset = - pos[1] if pos[1] < 0 else 0
    pos_13_offset = shape[-1] - (pos[1] + pos[3]) if pos[1] + pos[3] > shape[-1] else 0

    patch_core = patch[..., pos_0_offset: pos[2] + pos_02_offset, pos_1_offset: pos[3] + pos_13_offset]
    patch_core_g0 = patch_core > 0
    if len(patch_core_g0.shape) == 3:
        patch_core_g0 = patch_core_g0.any(0)
    
    tensor_set = tensor[..., pos[0] + pos_0_offset: pos[0] + pos[2] + pos_02_offset,
                            pos[1] + pos_1_offset: pos[1] + pos[3] + pos_13_offset]
    if patch_core.nelement() != 0 and patch_core[..., patch_core_g0].nelement() != 0 and \
        tensor_set.nelement() != 0:
        if value is None:
            tensor_set[..., patch_core_g0] = patch_core[..., patch_core_g0].type(tensor.dtype)
        else:
            value = value.type(tensor.dtype) if isinstance(value, torch.Tensor) else value
            tensor_set[..., patch_core_g0] = value
    return tensor


def classify_concept(tensor):
    """
    Args:
        Tensor: shape [H, W] and must have one of concept type of "Line", "RectSolid", "Rect", "Lshape".
    
    Returns:
        tensor_type: returns one of "Line", "RectSolid", "Rect", "Lshape".
    """
    assert len(tensor.shape) == 2
    tensor = shrink(tensor)[0]
    shape = tensor.shape
    assert set(np.unique(to_np_array(tensor.flatten())).tolist()).issubset({0., 1.})

    if shape[0] == 1 or shape[1] == 1:
        assert tensor.all()
        tensor_type = "Line"
    elif tensor.all():
        tensor_type = "RectSolid"
    elif tensor[0,0] == 1 and tensor[-1,0] == 1 and tensor[0,-1] == 1 and tensor[-1,-1] == 1:
        tensor_revert = 1 - tensor
        tensor_revert_shrink = shrink(tensor_revert)[0]
        assert tensor_revert_shrink.shape[0] == shape[0] - 2
        assert tensor_revert_shrink.shape[1] == shape[1] - 2
        tensor_type = "Rect"
    elif (tensor[0,0] == 1).long() + (tensor[-1,0] == 1).long() + (tensor[0,-1] == 1).long() + (tensor[-1,-1] == 1).long() == 3:
        tensor_type = "Lshape"
    else:
        assert (tensor == 0).all()
        tensor_type = None
    return tensor_type


def get_pos_intersection(pos1, pos2):
    """Get intersection of the position."""
    pos = [max(pos1[0], pos2[0]),
           max(pos1[1], pos2[1]),
           min(pos1[0] + pos1[2], pos2[0] + pos2[2]) - max(pos1[0], pos2[0]),
           min(pos1[1] + pos1[3], pos2[1] + pos2[3]) - max(pos1[1], pos2[1]),
          ]
    if pos[2] > 0 and pos[3] > 0:
        return pos
    else:
        return None


def get_obj_from_mask(input, obj_mask=None):
    """Get the object from the mask."""
    if obj_mask is None:
        return input
    assert input.shape[-2:] == obj_mask.shape
    if isinstance(input, np.ndarray):
        input = torch.FloatTensor(input)
    if isinstance(obj_mask, np.ndarray):
        obj_mask = torch.BoolTensor(obj_mask.astype(bool))
    shape = input.shape
    if len(shape) == 3:
        output = torch.zeros_like(input).reshape(input.shape[0], -1)
        idx = obj_mask.flatten().bool()
        output[:, idx] = input.reshape(input.shape[0], -1)[:, idx]
    else:
        output = torch.zeros_like(input).flatten()
        idx = obj_mask.flatten().bool()
        output[idx] = input.flatten()[idx]
    return output.reshape(shape)


def find_connected_components(input, is_diag=True, is_mask=False):
    """Find all the connected components, regardless of color."""
    input = to_np_array(input)
    shape = input.shape
    if is_diag:
        structure = [[1,1,1], [1,1,1], [1,1,1]]
    else:
        structure = [[0,1,0], [1,1,1], [0,1,0]]
    if len(shape) == 3:
        input_core = input.mean(0)
    else:
        input_core = input
    labeled, ncomponents = ndimage.measurements.label(input_core, structure)

    objects = []
    for i in range(1, ncomponents + 1):
        obj_mask = (labeled == i).astype(int)
        obj = shrink(get_obj_from_mask(input, obj_mask))
        if is_mask:
            objects.append(obj + (obj_mask,))
        else:
            objects.append(obj)
    return objects


def shrink(input):
    """ Find the smallest region of your matrix that contains all the nonzero elements """
    if not isinstance(input, torch.Tensor):
        input = torch.FloatTensor(input)
        is_numpy = True
    else:
        is_numpy = False
    if input.abs().sum() == 0:
        return input, (0, 0, input.shape[-2], input.shape[-1])
    if len(input.shape) == 3:
        input_core = input.mean(0)
    else:
        input_core = input
    rows = torch.any(input_core.bool(), axis=-1)
    cols = torch.any(input_core.bool(), axis=-2)
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    shrinked = input[..., ymin:ymax+1, xmin:xmax+1]
    pos = (ymin.item(), xmin.item(), shrinked.shape[-2], shrinked.shape[-1])
    if is_numpy:
        shrinked = to_np_array(shrinked)
    return shrinked, pos


def find_connected_components_colordiff(input, is_diag=True, color=True, is_mask=False):
    """
    Find all the connected components, considering color.
    
    :param input: Input tensor of shape (10, H, W)
    :param is_diag: whether or not diagonal connections should be considered
    as part of the same object.
    :param color: whether or not to divide by color of each object.
    :returns: list of tuples of the form (row_i, col_i, rows, cols)
    """
    input = to_np_array(input)
    shape = input.shape

    if len(shape) == 3:
        assert shape[0] == 3
        color_list = np.unique(input.reshape(shape[0], -1), axis=-1).T
        bg_color = np.zeros(shape[0])
    else:
        input_core = input
        color_list = np.unique(input)
        bg_color = 0

    objects = []
    for c in color_list:
        if not (c == bg_color).all():
            if len(shape) == 3:
                mask = np.array(input!=c[:,None,None]).any(0, keepdims=True).repeat(shape[0], axis=0)
            else:
                mask = np.array(input!=c, dtype=int)
            color_mask = np.ma.masked_array(input, mask)
            color_mask = color_mask.filled(fill_value=0)
            objs = find_connected_components(color_mask, is_diag=is_diag, is_mask=is_mask)
            objects += objs
    return objects


def mask_iou_score(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> float:
    """
    Computes the IoU (Intersection over Union) score between two masks.

    Args:
        pred_mask: tensor of shape [..., 1, H, W], where each value is either 0 or 1.
        target_mask: tensor [..., 1, H, W], where each value is either 0 or 1.

    Returns:
        The IoU score, wish shape [...], in the range [0, 1].
    """
    pred_mask = pred_mask.round() if torch.is_floating_point(pred_mask) else pred_mask
    return torch.sum(torch.logical_and(pred_mask, target_mask), dim=(-3, -2, -1)) / \
            torch.sum(torch.logical_or(pred_mask, target_mask), dim=(-3, -2, -1)).clamp(min=1)


def score_fun_IoU(pred,
                  target,
                  exclude: torch.Tensor=None):
    """
    Obtain the matching score between two arbitrary shaped 2D matrices.

    The final score is the number of matched elements for all common patches,
    divided by height_max * width_max that use the maximum dimension of pred and target.
    This is in similar spirit as IoU (Intersection over Union).
    """

    # Obtain the value the Concept graph is holding:
    if not isinstance(pred, torch.Tensor) and not isinstance(pred, np.ndarray):
        pred = pred.get_root_value()
    if not isinstance(target, torch.Tensor) and not isinstance(target, np.ndarray):
        target = target.get_root_value()
    
    assert isinstance(pred, torch.Tensor)
    assert isinstance(target, torch.Tensor)

    pred_size_compare = None

    if not (len(pred.shape) == len(target.shape) == 2):
        # Pred and target are not both 2D matrices:
        return None, {"error": "Pred and target are not both 2D matrices."}
    
    excluded = 0
    if exclude is not None:
        excluded = torch.logical_and(pred == exclude, target == exclude).sum().item()

    # Find which matrix is larger and which is smaller

    if pred.shape[0] <= target.shape[0] and pred.shape[1] <= target.shape[1] or pred.shape[0] > target.shape[0] and pred.shape[1] > target.shape[1]:
        if pred.shape[0] <= target.shape[0] and pred.shape[1] <= target.shape[1]:
            pred_size_compare = ("smaller", "smaller")
            patch_s, patch_l = pred, target
 
        elif pred.shape[0] > target.shape[0] and pred.shape[1] > target.shape[1]:
            pred_size_compare = ("larger", "larger")
            patch_s, patch_l = target, pred

        shape_s = patch_s.shape
        shape_l = patch_l.shape
        best_idx = None
        max_score = -1
        for i in range(shape_l[0] - shape_s[0] + 1):
            for j in range(shape_l[1] - shape_s[1] + 1):
                patch_l_chosen = patch_l[i: i + shape_s[0], j: j + shape_s[1]]
                score = masked_equal(patch_l_chosen, patch_s, exclude=exclude).sum()
                if score > max_score:
                    best_idx = (i, j)
                    max_score = score
        final_score = max_score.float() / (shape_l[0] * shape_l[1] - excluded)
    else:
        if pred.shape[0] <= target.shape[0] and pred.shape[1] > target.shape[1]:
            pred_size_compare = ("smaller", "larger")
            height_s, height_l = pred.shape[0], target.shape[0]
            width_s, width_l = target.shape[1], pred.shape[1]
            pred_height_smaller = True
        else:
            pred_size_compare = ("larger", "smaller")
            height_l, height_s = pred.shape[0], target.shape[0]
            width_l, width_s = target.shape[1], pred.shape[1]
            pred_height_smaller = False
        best_idx = None
        max_score = -1
        for i in range(height_l - height_s + 1):
            for j in range(width_l - width_s + 1):
                if pred_height_smaller:
                    pred_chosen = pred[:, j: j + width_s]
                    target_chosen = target[i: i + height_s, :]
                else:
                    pred_chosen = pred[i: i + height_s, :]
                    target_chosen = target[:, j: j + width_s]

                score = masked_equal(pred_chosen, target_chosen, exclude=exclude).sum()
                if score > max_score:
                    best_idx = (i, j)
                    max_score = score
        final_score = max_score.float() / (height_l * width_l - excluded)
    info = {"best_idx": best_idx,
            "pred_size_compare": pred_size_compare}
    return final_score, info


def check_result_true(results):
    """Check if there exists a node where the result is True for all examples."""
    is_result_true = False
    node_true = None
    for node_key, result in results.items():
        value_list = []
        for example_key, value in result.items():
            if not isinstance(value, torch.Tensor):
                value = value.get_root_value()
            if isinstance(value, torch.BoolTensor) and tuple(value.shape) == ():
                value_list.append(value)
        if len(value_list) > 0:
            is_result_true = to_np_array(torch.stack(value_list).all())
            if is_result_true:
                node_true = node_key
                break
    return is_result_true, node_true


def get_obj_bounding_pos(objs):
    """Get the pos for the bounding box for a dictionary of objects."""
    row_min = np.Inf
    row_max = -np.Inf
    col_min = np.Inf
    col_max = -np.Inf
    for obj_name, obj in objs.items():
        pos = obj.get_node_value("pos")
        if pos[0] < row_min:
            row_min = int(pos[0])
        if pos[0] + pos[2] > row_max:
            row_max = int(pos[0] + pos[2])
        if pos[1] < col_min:
            col_min = int(pos[1])
        if pos[1] + pos[3] > col_max:
            col_max = int(pos[1] + pos[3])
    pos_bounding = (row_min, col_min, row_max - row_min, col_max - col_min)
    return pos_bounding


def get_comp_obj(obj_dict, CONCEPTS):
    """Get composite object from multiple objects."""
    pos_bounding = get_obj_bounding_pos(obj_dict)
    comp_obj = CONCEPTS["Image"].copy().set_node_value(torch.zeros(int(pos_bounding[2]), int(pos_bounding[3])))
    for obj_name, obj in obj_dict.items():
        obj_copy = obj.copy()
        pos_obj = obj_copy.get_node_value("pos")
        obj_copy.set_node_value([pos_obj[0] - pos_bounding[0], pos_obj[1] - pos_bounding[1], pos_obj[2], pos_obj[3]], "pos")
        comp_obj.add_obj(obj_copy, obj_name=obj_name, change_root=True)
    comp_obj.set_node_value([0, 0, int(pos_bounding[2]), int(pos_bounding[3])], "pos")
    return comp_obj, pos_bounding


def get_indices(tensor, pos=None, includes_neighbor=False, includes_self=True):
    """Get the indices of nonzero elements of an image.

    Args:
        tensor: 2D or 3D tensor. If 3D, it must have the shape of [C, H, W] where C is the number of channels.
        pos: position of the upper-left corner pixel of the tensor in the larger image. If None, will default as (0, 0).
        includes_neighbor: whether to include indices of neighbors (up, down, left, right).
        includes_self: if includes_neighbor is True, whether to include its own indices.

    Returns:
        indices: a list of indices satisfying the specification.
    """
    mask = tensor > 0
    if len(mask.shape) == 3:
        mask = mask.any(0)
    pos_add = (int(pos[0]), int(pos[1]))  if pos is not None else (0, 0)
    indices = []
    self_indices = []
    for i, j in torch.stack(torch.where(mask)).T:
        i, j = int(i) + pos_add[0], int(j) + pos_add[1]
        self_indices.append((i, j))
        if includes_neighbor:
            indices.append((i + 1, j))
            indices.append((i - 1, j))
            indices.append((i, j + 1))
            indices.append((i, j - 1))
    if includes_neighbor:
        if not includes_self:
            indices = list(set(indices).difference(set(self_indices)))
        else:
            indices = remove_duplicates(indices)
    else:
        indices = self_indices
    return indices


def get_op_type(op_name):
    """Get the type of the op.

    Node naming convention:
        op:          operator, e.g. "Draw"
        op-in:       operator's input and output nodes, e.g. "Draw-1:Image"
        op-attr:     attribute from an op output, e.g. "Draw-o^color:Image", "input-1^pos:Pos"
        op-sc:       a concept node belonging to an operator's input's selector, e.g. "Draw-1->sc$obj_0:c0"
        op-so:       a relation node belonging to an operator's input's selector, e.g. "Draw-1->so$(obj_0:c0,obj_1:c1):r1"
        op-op:       operator's inner operator, e.g. "ForGraph->op$Copy"
        input:       input_placeholder_nodes, e.g. "input-1:Image"
        concept:     constant concept node, e.g. "concept-1:Image"
        o:           operator definition node, e.g. "o$Draw"
        c:           concept definition node, e.g. "c$Image"
        result:      input or intermediate nodes, e.g. "result$Identity->0:Color", "result$Draw->0->obj_1:Image" (The obj_1 at the 0th example at the outnode of "Draw")
        target:      target nodes, e.g. "target$0:Color", "target$1->obj_1:Image" (The obj_1 at the 1th example at the target)
        opparse:     parsed op, e.g. "opparse$Draw->0->obj_1->RotateA"

    Returns:
        op_type: string indicating the type of the node.
    """
    if isinstance(op_name, tuple) or isinstance(op_name, list):
        op_types = tuple([get_op_type(op_name_ele) for op_name_ele in op_name])
        return op_types
    elif isinstance(op_name, dict) or isinstance(op_name, set):
        raise Exception("op_name can only be a tuple, a list or a string!")
    if op_name.startswith("target$"):
        op_type = "target"
    elif "->" in op_name:
        if op_name.startswith("result$"):
            op_type = "result"
        elif op_name.startswith("opparse$"):
            op_type = "opparse"
        else:
            op_name_core = op_name.split("->")[-1]
            type_name, op_name_core = op_name_core.split("$")
            if type_name == "sc":
                op_type = "op-sc"
            elif type_name == "so":
                op_type = "op-so"
            elif type_name == "op":
                op_type = "op-op"
            else:
                raise
    else:
        if "^" in op_name:
            op_type = "op-attr"
        else:
            if "input" in op_name:
                op_type = "input"
            elif "concept" in op_name:
                op_type = "concept"
            elif op_name.startswith("o$"):
                op_type = "o"
            elif op_name.startswith("c$"):
                op_type = "c"
            elif ":" in op_name:
                if "-o" in op_name:
                    raise Exception("{} is not a valid node for node_type!".format(op_name))
                else:
                    op_type = "op-in"
            else:
                op_type = "op"
    return op_type


def get_edge_path(path):
    """Get the edge_path, where path is e.g. 
        [('target$0->obj_4:Image', 'N-target-parentResult', 'result$Draw->0->obj_2:Image'),
         ('result$Draw->0->obj_2:Image', 'N-result-parent', 'Draw'),
         ('Draw', 'N-op-result', 'result$Draw->3:Image')], and will return
        ('N-target-parentResult', 'N-result-parent', 'N-op-result').
    """
    return tuple([ele[1] for ele in path])


def get_ids_same_value(logits, id, epsilon=1e-6):
    """Get the ids of that logits that has the same value as logits[id]."""
    return to_np_array(torch.where((logits - logits[id]).abs() < epsilon)[0], full_reduce=False).tolist()


def normalize_same_value(logits, is_normalize=True):
    """For repetitive numbers in logits, regard them only appear once."""
    if is_normalize:
        logits_value = logits.detach()
        same_value_array = (logits_value[:, None] == logits_value[None, :]).sum(0).float()
        return logits - torch.log(same_value_array)
    else:
        return logits


def get_normalized_entropy(dist, is_normalize=True):
    """Compute the entropy for probability with repetitive values."""
    entropy = dist.entropy()
    if is_normalize:
        logits_value = dist.logits.detach()
        same_value_array = (logits_value[:, None] == logits_value[None, :]).sum(0).float()
        entropy = entropy - (dist.probs * same_value_array.log()).sum()
    return entropy


def clip_grad(optimizer):
    """Clip gradient"""
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))


def get_module_parameters(List, device):
    """Get the learnable parameters in a dictionary, used for learning the embedding for OPERATORS, CONCEPTS, NEIGHBOR_EMBEDDING, etc."""
    parameters_dict = {}
    for item in List:
        for op_name, op in item.items():
            if not isinstance(op, torch.Tensor):
                op = op.get_node_repr()
            op.data = op.data.to(device)
            parameters_dict[op_name] = op
    return parameters_dict


def assign_embedding_value(embedding_parameters_dict, List):
    """Assign the values to the concept representations."""
    for op_name_save, value in embedding_parameters_dict.items():
        for Dict in List:
            for op_name, op in Dict.items():
                if op_name_save == op_name:
                    if isinstance(op, torch.Tensor):
                        op.data = torch.FloatTensor(value)
                    else:
                        op.get_node_repr().data = torch.FloatTensor(value)


def get_hashing(string_repr, length=None):
    """Get the hashing of a string."""
    import hashlib, base64
    hashing = base64.b64encode(hashlib.md5(string_repr.encode('utf-8')).digest()).decode().replace("/", "a")[:-2]
    if length is not None:
        hashing = hashing[:length]
    return hashing


def persist_hash(string_repr):
    import hashlib
    return int(hashlib.md5(string_repr.encode('utf-8')).hexdigest(), 16)


def tensor_to_string(tensor):
    """Transform a tensor into a string, for hashing purpose."""
    if tensor is None:
        return "None"
    return ",".join([str(ele) for ele in np.around(to_np_array(tensor), 4).flatten().tolist()])


def get_repr_dict(dct):
    """Assumes all elements of the dictionary have get_string_repr"""
    string = ""
    for key, item in dct.items():
        if isinstance(item, torch.Tensor):
            string += "({}){}".format(key, tensor_to_string(item))
        else:
            string += "({}){}".format(key, item.get_string_repr())
    return string

def check_same_tensor(List):
    """Returns True if all the element tensors of the List have the same shape and value."""
    element_0 = List[0]
    is_same = True
    for element in List[1:]:
        if tuple(element.shape) != tuple(element_0.shape) or not (element == element_0).all():
            is_same = False
            break
    return is_same


def repeat_n(*args, **kwargs):
    List = []
    for tensor in args:
        if tensor is None:
            result = None
        elif isinstance(tensor, tuple):
            result = tuple([repeat_n(ele, **kwargs) for ele in tensor])
        else:
            result = tensor.repeat(kwargs["n_repeats"], *torch.ones(len(tensor.shape)-1).long())
        List.append(result)
    if len(args) > 1:
        return tuple(List)
    else:
        return List[0]


def identity_fun(input):
    return input


class Task_Dict(dict):
    """
    A dictionary storing the buffer for the top K solution for each task. It has the following structure:

    {task_hash1: {
        TopKList([
            {"graph_hash": graph_hash,
             "score": score,
             "graph_state": graph_state,
             "action_record": action_record,
            }
        ]),
     task_hash2: {...},
     ...
    }
    """
    def __init__(self, K, mode="max"):
        self.K = K
        self.mode = mode

    def init_task(self, task_hash):
        """Initialize a new task."""
        assert task_hash not in self
        self[task_hash] = TopKList(K=self.K, sort_key="score", duplicate_key="graph_hash", mode=self.mode)

    def reset_task(self, task_hash):
        """Reset an existing task or initialize a new task."""
        self[task_hash] = TopKList(K=self.K, sort_key="score", duplicate_key="graph_hash", mode=self.mode)

    @property
    def n_examples_all(self):
        """Return a dictionary of number of examples in a task."""
        return {task_hash: len(top_k_list) for task_hash, top_k_list in self.items()}

    @property
    def n_examples_per_task(self):
        """Get average number of examples per task."""
        return np.mean(list(self.n_examples_all.values()))

    @property
    def std_examples_per_task(self):
        """Get average number of examples per task."""
        return np.std(list(self.n_examples_all.values()))

    @property
    def score_all(self):
        return {task_hash: np.mean([element["score"] for element in top_k_list]) for task_hash, top_k_list in self.items()}

    @property
    def mean_score(self):
        return np.mean(list(self.score_all.values()))

    @property
    def std_score(self):
        return np.std(list(self.score_all.values()))

    def update_score_with_ebm_dict(self, ebm_dict, cache_forward=True, is_update_score=True):
        """Update the score using the new ebm_dict."""
        self.ebm_dict = ebm_dict
        for task_hash, task_top_k_list in self.items():
            for element in task_top_k_list:
                element["graph_state"].set_ebm_dict(ebm_dict)
                element["graph_state"].set_cache_forward(False)  # Important: clear the cache
                if is_update_score:
                    element["score"] = element["graph_state"].get_score()
                if cache_forward:
                    element["graph_state"].set_cache_forward(True)
        return self

    def to(self, device):
        """Move all modules to device."""
        self.ebm_dict.to(device)
        for task_hash, task_top_k_list in self.items():
            for element in task_top_k_list:
                if "graph_state" in element:
                    element["graph_state"].to(device)
                if "selector" in element:
                    element["selector"] = element["selector"].to(device)
        return self


class Combined_Dict(dict):
    """Dictionary holding the EBMs. The parameters of the EBMs can is independent."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_ebm_share_param = False

    def set_is_relation_z(self, is_relation_z):
        self.is_relation_z = is_relation_z
        return self

    def parameters(self):
        return itertools.chain.from_iterable([ebm.parameters() for key, ebm in self.items()])

    def to(self, device):
        """Move all modules to device."""
        for key in self:
            self[key] = self[key].to(device)
        return self

    @property
    def model_dict(self):
        """Returns the model_dict for each model."""
        return {key: model.model_dict for key, model in self.items()}


def get_instance_keys(class_instance):
    """Get the instance keys of a class"""
    return [key for key in vars(class_instance) if key[:1] != "_"]


class Model_Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        assert model.__class__.__name__ == "ConceptEBM"
        self.model = model

    def set_c(self, c_repr):
        self.c_repr = c_repr
        return self

    def forward(self, *args, **kwargs):
        kwargs["c_repr"] = self.c_repr
        return self.model.forward(*args, **kwargs)

    def classify(self, *args, **kwargs):
        return self.model.classify(*args, **kwargs)

    def ground(self, *args, **kwargs):
        return self.model.ground(*args, **kwargs)

    def state_dict(self):
        return self.model.state_dict()

    def __getattribute__(self, item):
        """Obtain the attributes. Prioritize the instance attributes in self.model."""
        if item == "model":
            return object.__getattribute__(self, "model")
        elif item.startswith("_"):
            return object.__getattribute__(self, item)
        elif item != "c_repr" and item in get_instance_keys(self.model):
            return getattr(self.model, item)
        else:
            return object.__getattribute__(self, item)

    @property
    def model_dict(self):
        return self.model.model_dict


class Shared_Param_Dict(nn.Module):
    """Dictionary holding the EBMs. The parameters of the EBMs can is shared."""
    def __init__(
        self,
        concept_model=None,
        relation_model=None,
        concept_repr_dict=None,
        relation_repr_dict=None,
        is_relation_z=True,
    ):
        super().__init__()
        self.concept_model = concept_model
        self.relation_model = relation_model
        self.concept_repr_dict = {}
        if concept_repr_dict is not None:
            for key, item in concept_repr_dict.items():
                self.concept_repr_dict[key] = torch.FloatTensor(item) if not isinstance(item, torch.Tensor) else item
        self.relation_repr_dict = {}
        if relation_repr_dict is not None:
            for key, item in relation_repr_dict.items():
                self.relation_repr_dict[key] = torch.FloatTensor(item) if not isinstance(item, torch.Tensor) else item
        self.is_ebm_share_param = True
        self.is_relation_z = is_relation_z

    def set_is_relation_z(self, is_relation_z):
        self.is_relation_z = is_relation_z
        return self

    def add_c_repr(self, c_repr, c_str, ebm_mode):
        if ebm_mode == "concept":
            self.concept_repr_dict[c_str] = c_repr
        elif ebm_mode == "operator":
            self.relation_repr_dict[c_str] = c_repr
        else:
            raise Exception("ebm_mode {} is not valid!".format(ebm_mode))
        return self

    def update(self, new_dict):
        assert new_dict.__class__.__name__ == "Shared_Param_Dict"
        if new_dict.concept_model is not None:
            self.concept_model = new_dict.concept_model
        if new_dict.relation_model is not None:
            self.relation_model = new_dict.relation_model
        for key, item in new_dict.concept_repr_dict.items():
            if key in self:
                assert (item == self.concept_repr_dict[key]).all()
            self.concept_repr_dict[key] = item
        for key, item in new_dict.relation_repr_dict.items():
            if key in self:
                assert (item == self.relation_repr_dict[key]).all()
            self.relation_repr_dict[key] = item
        return self

    def __setitem__(self, key, model):
        assert model.__class__.__name__ == "ConceptEBM"
        if model.mode == "concept":
            assert self.concept_model is None
            self.concept_model = model
            self.concept_repr_dict[model.c_str] = model.c_repr
        elif model.mode == "operator":
            assert self.relation_model is None
            self.relation_model = model
            self.relation_repr_dict[model.c_str] = model.c_repr
        else:
            raise Exception("ebm_mode '{}' is not valid!".format(model.mode))

    def __getitem__(self, key):
        if key in self.concept_repr_dict:
            return Model_Wrapper(self.concept_model).set_c(c_repr=self.concept_repr_dict[key])
        elif key in self.relation_repr_dict:
            return Model_Wrapper(self.relation_model).set_c(c_repr=self.relation_repr_dict[key])
        else:
            raise Exception("key '{}' not in concept_repr_dict nor relation_repr_dict.".format(key))

    def keys(self):
        return list(self.concept_repr_dict.keys()) + list(self.relation_repr_dict.keys())

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        return item in self.keys()

    def has_key(self, k):
        return k in self.keys()

    def is_model_exist(self, ebm_mode):
        if ebm_mode == "concept":
            return self.concept_model is not None
        elif ebm_mode == "operator":
            return self.relation_model is not None
        else:
            raise Exception("ebm_mode {} is not valid!".format(ebm_mode))

    def parameters(self):
        iterables = []
        if self.concept_model is not None:
            iterables.append(self.concept_model.parameters())
        if self.relation_model is not None:
            iterables.append(self.relation_model.parameters())
        return itertools.chain.from_iterable(iterables)

    def to(self, device):
        """Move all modules to device."""
        if self.concept_model is not None:
            self.concept_model.to(device)
        if self.relation_model is not None:
            self.relation_model.to(device)
        for key in self.concept_repr_dict:
            self.concept_repr_dict[key] = self.concept_repr_dict[key].to(device)
        for key in self.relation_repr_dict:
            self.relation_repr_dict[key] = self.relation_repr_dict[key].to(device)
        return self

    @property
    def model_dict(self):
        model_dict = {"type": "Shared_Param_Dict"}
        model_dict["concept_model_dict"] = self.concept_model.model_dict if self.concept_model is not None else None
        model_dict["relation_model_dict"] = self.relation_model.model_dict if self.relation_model is not None else None
        model_dict["concept_repr_dict"] = {key: to_np_array(item) for key, item in self.concept_repr_dict.items()}
        model_dict["relation_repr_dict"] = {key: to_np_array(item) for key, item in self.relation_repr_dict.items()}
        return model_dict


def get_str_value(string_to_split, string):
    string_splited = string_to_split.split("-")
    if string in string_splited:
        return eval(string_splited[string_splited.index(string)+1])
    else:
        return None

def is_diagnose(loc, filename):
    """If the given loc and filename matches that of the diagose.yml, will return True and (later) call an pde.set_trace()."""
    try:
        with open(get_root_dir() + "/experiments/diagnose.yml", "r") as f:
            Dict = yaml.load(f, Loader=yaml.FullLoader)
    except:
        return False
    if Dict is None:
        return False
    Dict.pop(None, None)
    if not ("loc" in Dict and "dirname" in Dict and "filename" in Dict):
        return False
    if loc == Dict["loc"] and filename == Dict["dirname"] + Dict["filename"]:
        return True
    else:
        return False


def model_parallel(model, args):
    if args.parallel_mode == "None":
        device = get_device(args)
        model.to(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            if args.parallel_mode == "ddp":
                if args.local_rank >= 0:
                    torch.cuda.set_device(args.local_rank) 
                    device = torch.device(f"cuda:{args.local_rank}")
                torch.distributed.init_process_group(backend='nccl', rank=args.local_rank, world_size=torch.cuda.device_count())
                torch.set_num_threads(os.cpu_count()//(torch.cuda.device_count() * 2))
                model.to(device)
                model = torch.nn.parallel.DistributedDataParallel(model)
            elif args.parallel_mode == "dp":
                model = MineDataParallel(model)
                model.to(device)
    return model, device


class MineDataParallel(nn.parallel.DataParallel):
    def __getattribute__(self, key):
        module_attrs = [
            'training',
            'mode',
            'in_channels',
            'repr_dim',
            'w_type',
            'w_dim',
            'mask_mode',
            'channel_base',
            'two_branch_mode',
            'mask_arity',
            'is_spec_norm',
            'is_res',
            'c_repr_mode',
            'c_repr_first',
            'c_repr_base',
            'z_mode',
            'z_first',
            'z_dim',
            'pos_embed_mode',
            'aggr_mode',
            'img_dims',
            'act_name',
            'normalization_type',
            'dropout',
            'self_attn_mode',
            'last_act_name',
            'n_avg_pool',
            'model_dict',
        ]
        if key in module_attrs:
            return object.__getattribute__(self.module, key)
        else:
            if hasattr(MineDataParallel, key):
                return object.__getattribute__(self, key)
            else:
                return super().__getattribute__(key)
