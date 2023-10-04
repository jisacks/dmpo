import torch
import torch.nn as nn

from ..envs.quadrotor import QuadrotorEnv
from ..mpc.task.base_rollout_task import BaseRolloutTask
from ..mpc.task.quadrotor_rollout_task import QuadrotorRolloutTask
from ..controllers.dmpo_policy import DMPOPolicy
from .. import utils

from torch.nn import Module
from typing import Dict, Any, Optional, List, Tuple, Callable

ModelTypes = ['dmpo_policy']
EnvTypes = ['quadrotor']

def create_model(model_type: str,
                 tensor_args: Dict[str, Any] = {'device': 'cpu', 'dtype': torch.float},
                 **kwargs: Dict[str, Any]) -> Module:
    if model_type in ModelTypes:
        if model_type == 'dmpo_policy':
            return DMPOPolicy(tensor_args=tensor_args, **kwargs)
    else:
        raise ValueError('Invalid model type {} specified.'.format(model_type))

def create_env(env_name: str,
               env_config_file: Optional[str]=None,
               tensor_args: Dict[str, Any]={'device': 'cpu', 'dtype': torch.float},
               **kwargs: Dict[str, Any]):
    if env_name in EnvTypes:
        # Load in the environment configuration file
        if not env_config_file is None:
            env_params = utils.load_yaml(env_config_file)

            # Allow for kwargs to overwrite configuration file
            for k, v in env_params.items():
                if not k in kwargs.keys():
                    kwargs[k] = v

        # Handle each environment separately
        if env_name == 'quadrotor':
            return QuadrotorEnv(tensor_args=tensor_args, **kwargs)
    else:
        raise ValueError('Specified environment type {} is unsupported.'.format(env_name))

def create_task(env_name: str,
                num_envs: int,
                task_config_file: str,
                tensor_args: Dict[str, Any] = {'device': 'cpu', 'dtype': torch.float},
                **kwargs: Dict[str, Any]) -> BaseRolloutTask:
    if env_name in EnvTypes:
        # Load in the task configuration file
        task_params = utils.load_yaml(task_config_file)

        # Handle each environment separately
        if env_name == 'quadrotor':
            return QuadrotorRolloutTask(exp_params=task_params,
                                        num_envs=num_envs,
                                        tensor_args=tensor_args,
                                        **kwargs)
    else:
        raise ValueError('Specified environment type {} is unsupported.'.format(env_name))

def create_optim(config: Dict[str, Any]) -> Tuple[Callable, Dict[str, Any]]:
    optim_type = config.get('optim', 'Adam')
    optim_args = config.get('optim_args', {})

    if optim_type == 'Adam':
        optim = torch.optim.Adam
    elif optim_type == 'SGD':
        optim = torch.optim.SGD
    elif optim_type == 'Adagrad':
        optim = torch.optim.Adagrad
    elif optim_type == 'RMSprop':
        optim = torch.optim.RMSprop
    elif optim_type == 'RAdam':
        optim = torch.optim.RAdam
    else:
        raise ValueError('Specified optimizer type {} unsupported.'.format(optim_type))

    return optim, optim_args