import torch

from .base_rollout_task import BaseRolloutTask
from ..model.quadrotor_model import QuadrotorModel
from ..rollout.quadrotor_rollout import QuadrotorRollout

from typing import Optional, Dict, Any

class QuadrotorRolloutTask(BaseRolloutTask):
    def __init__(self,
                 exp_params: Dict[str, Any],
                 num_envs: Optional[int]=None,
                 tensor_args: Dict[str, Any]={'device': "cpu", 'dtype': torch.float32},
                 **kwargs) -> None:
        super().__init__(
            exp_params=exp_params,
            num_envs=num_envs,
            tensor_args=tensor_args
        )

    def get_rollout_fn(self, exp_params: Dict[str, Any]):
        dynamics_params = exp_params['model']
        self.dynamics_model = QuadrotorModel(tensor_args=self.tensor_args, num_envs=self.num_envs, **dynamics_params)
        rollout_fn = QuadrotorRollout(dynamics_model=self.dynamics_model,
                                      tensor_args=self.tensor_args,
                                      num_envs=self.num_envs,
                                      exp_params=exp_params)
        return rollout_fn
