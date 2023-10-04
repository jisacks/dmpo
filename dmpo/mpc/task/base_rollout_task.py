import torch

from ..rollout.rollout_generator import RolloutGenerator

from torch import Tensor
from typing import Optional, Dict, Any, List

class BaseRolloutTask():
    """
    Base class for defining MPC tasks
    """
    def __init__(self,
                 exp_params: Dict[str, Any],
                 num_envs: Optional[int]=None,
                 tensor_args: Dict[str, Any]={'device': "cpu", 'dtype': torch.float32}) -> None:
        """
        :param exp_params: Rollout parameters
        :param num_envs: # of environments (threads)
        :param tensor_args: PyTorch params
        """
        self.tensor_args = tensor_args
        self.num_envs = num_envs
        self.running: bool = False
        self.top_idx: Optional[List[Tensor]] = None
        self.top_values: Optional[List[Tensor]] = None
        self.top_trajs: Optional[List[Tensor]] = None

        self.rollout_fn = self.init_rollout(exp_params)
        self.init_aux(exp_params, num_envs)

    def init_aux(self, exp_params: Dict[str, Any], num_envs: int):
        pass

    def get_rollout_fn(self, exp_params: Dict[str, Any]):
        raise NotImplementedError('Function get_rollout_fn has not implemented.')

    def init_rollout(self, exp_params):
        rollout_fn = self.get_rollout_fn(exp_params)
        return RolloutGenerator(rollout_fn=rollout_fn, tensor_args=self.tensor_args)

    def run_rollouts(self, states: Tensor, act_seqs: Tensor) -> Dict[str, Any]:
        trajectories = self.rollout_fn.run_rollouts(states, act_seqs)
        return trajectories

    def update_params(self, kwargs: Dict[str, Any]) -> bool:
        self.rollout_fn.update_params(kwargs)
        return True
