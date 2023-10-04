import torch
import torch.nn as nn
import numpy as np
import os
import copy
import yaml
import ghalton

from torch import Tensor
from numpy.typing import NDArray
from typing import Optional, Tuple, Dict, Any, Union, List

ActivationType = {'relu': nn.ReLU,
                  'sigmoid': nn.Sigmoid,
                  'tanh': nn.Tanh,
                  'elu': nn.ELU,
                  'leakyrelu': nn.LeakyReLU,
                  'prelu': nn.PReLU,
                  'relu6': nn.ReLU6,
                  'selu': nn.SELU,
                  'celu': nn.CELU,
                  'gelu': nn.GELU,
                  'silu': nn.SiLU,
                  'mish': nn.Mish,
                  'softplus': nn.Softplus,
                  'tanhshrink': nn.Tanhshrink
                  }
TorchDtype = {'float': torch.float, 'double': torch.double, 'long': torch.long}

def compute_statistics(success_dict: Dict[str, Any]) -> Dict[str, Any]:
    successes = success_dict['successes']
    total_costs = success_dict['total_costs']

    mean_success_cost = torch.mean(total_costs[successes])
    std_success_cost = torch.std(total_costs[successes])
    mean_fail_cost = torch.mean(total_costs[~successes])
    std_fail_cost = torch.std(total_costs[~successes])
    mean_cost = torch.mean(total_costs)
    std_cost = torch.std(total_costs)
    ret_dict = dict(
        mean_success_cost=mean_success_cost,
        std_success_cost=std_success_cost,
        mean_fail_cost=mean_fail_cost,
        std_fail_cost=std_fail_cost,
        mean_cost=mean_cost,
        std_cost=std_cost
    )
    return ret_dict

def make_dir(path: str):
    """
    Create a directory if the given path does not exist
    """
    if not os.path.exists(path):
        os.makedirs(path)

def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)

def load_yaml(filename: str) -> Dict[str, Any]:
    with open(filename) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return params

def to_tensor(input: Union[List[Union[Tensor, NDArray, float]], Tensor, NDArray, float],
              tensor_args={'device':'cpu', 'dtype':torch.float32}) -> Tensor:
    if isinstance(input, list):
        if isinstance(input[0], list):
            return torch.tensor(input, **tensor_args)
        elif isinstance(input[0], np.ndarray):
            return torch.stack([to_tensor(x) for x in input])
        elif isinstance(input[0], torch.Tensor):
            return torch.stack(input).to(**tensor_args)
        elif isinstance(input[0], float):
            return torch.tensor(input, **tensor_args)
        else:
            raise ValueError('Invalid input to convert to tensor.')
    elif isinstance(input, torch.Tensor):
        return input.to(**tensor_args)
    elif isinstance(input, np.ndarray):
        return torch.from_numpy(input).to(**tensor_args)
    elif isinstance(input, float):
        return torch.tensor(input, **tensor_args)
    else:
        raise ValueError('Invalid input to convert to tensor.')

def as_list(x: Union[List[Any], Any], n: int =1) -> List[Any]:
    """
    Return a variable as a list
    :param x: Variable to be copied n times in a list (returns x if already a list)
    :param n: # of times to copy the variable
    """
    if isinstance(x, list):
        return x
    else:
        return [x for _ in range(n)]

def stack_list_tensors(input: List[Tensor]) -> List[Tensor]:
    """
    Converts a list (length T) of Tensors (N, ...) into a list (length N) of Tensors (T, ...)
    """
    out = []
    for idx in range(len(input[0])):
        tensor_list = torch.stack([x[idx] for x in input], dim=0)
        out.append(tensor_list)
    return out

def transpose_dict_list(input: List[Dict[str, Any]]) -> List[Dict[str, List[Any]]]:
    """
    Converts a list (length T) of dictionaries of lists\Tensors (length N) into a list (length N) of dictionaries
    consisting of lists (length T)
    """
    keys = list(input[0].keys())
    length = len(input[0][keys[0]])
    out = list({} for _ in range(length))
    for k in keys:
        y = [x[k] for x in input]
        for idx in range(length):
            v = [data[idx] if isinstance(data, list) or isinstance(data, torch.Tensor) else data for data in y]
            v = [x.detach() for x in v if isinstance(x, torch.Tensor)]
            out[idx][k] = copy.deepcopy(v)
    return out

def stack_list_list_dict_tensors(input: List[List[Dict[str, Tensor]]]) -> List[Dict[str, Tensor]]:
    """
    Converts a list (length T) of lists (length N) of dictionaries of Tensors to a list (length N) of dictionaries of
    Tensors (T, ...)
    """
    keys = list(input[0][0].keys())
    length = len(input[0])
    out = list({} for _ in range(length))
    for k in keys:
        for idx in range(length):
            out[idx][k] = []
            for t in range(len(input)):
                y = [x[k] for x in input[t]]
                out[idx][k].append(copy.deepcopy(y[idx]))
            out[idx][k] = torch.stack(out[idx][k])
    return out

def load_struct_from_dict(struct_instance, dict_instance):
    for key in dict_instance.keys():
        if (hasattr(struct_instance, key)):
            if (isinstance(dict_instance[key], dict)):
                sub_struct = load_struct_from_dict(getattr(struct_instance, key), dict_instance[key])
                setattr(struct_instance, key, sub_struct)
            else:
                setattr(struct_instance, key, dict_instance[key])
    return struct_instance

def set_if_empty(dict, key, value):
    if not key in dict.keys():
        dict[key] = value
    return dict

def generate_gaussian_halton_samples(num_samples: int,
                                     ndims: int,
                                     seed_val: int=123,
                                     device='cpu',
                                     dtype=torch.float32) -> Tensor:
    """
    Generate Halton sequence and transform to Gaussian distribution
    :param num_samples: # of samples
    :param ndims: # of independent Gaussians from which to sample
    :param seed_val: Seed for the Halton sequence generator
    :param device: PyTorch device
    :param dtype: PyTorch dtype
    """
    sequencer = ghalton.GeneralizedHalton(ndims, seed_val)
    uniform_halton_samples = torch.tensor(sequencer.get(num_samples), device=device, dtype=dtype)
    gaussian_halton_samples = torch.sqrt(torch.tensor([2.0], device=device, dtype=dtype)) \
                              * torch.erfinv(2 * uniform_halton_samples - 1)
    return gaussian_halton_samples