import torch

from torch import Tensor
from typing import Optional, Union, Dict, Any, Callable

class RolloutGenerator():
    """
    Class which handles rolling out dynamics models
    """
    def __init__(self,
                 rollout_fn: Optional[Callable[[Tensor, Tensor], dict]]=None,
                 tensor_args: Dict[str, Any]={'device': 'cpu', 'dtype': torch.float32}) -> None:
        self._rollout_fn = rollout_fn
        self.tensor_args = tensor_args

    @property
    def rollout_fn(self):
        return self._rollout_fn

    @rollout_fn.setter
    def rollout_fn(self, fn: Callable[[Tensor, Tensor], dict]):
        self._rollout_fn = fn

    def run_rollouts(self, state: Tensor, act_seq: Tensor) -> Dict[str, Any]:
        state = state.to(**self.tensor_args)
        act_seq = act_seq.to(**self.tensor_args)

        trajectories = self._rollout_fn(state, act_seq)
        return trajectories

    def update_params(self, kwargs: Dict[str, Any]) -> bool:
        return self.rollout_fn.update_params(**kwargs)