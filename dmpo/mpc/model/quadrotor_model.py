import torch
import numpy as np
import yaml

from ...envs.quadrotor_param import QuadrotorParam
from ...envs.math_utils import *

from torch import Tensor
from numpy.typing import NDArray
from typing import Dict, Any, Optional, List, Union, Tuple

class QuadrotorModel():
    '''
    Quadrotor model used by MPC
    '''
    def __init__(self,
                 config: str,
                 num_envs: int=1,
                 use_omega: bool = False,
                 action_is_mf: bool = True,
                 convert_mf_to_omega: bool = False,
                 use_delay_model: bool = False,
                 delay_coeff: float = 0.4,
                 tensor_args: Dict[str, Any]={'device': 'cpu', 'dtype': torch.float32}):
        """
        :param config: YAML configuration file, which will be parsed to form a QuadrotorParam object
        :param num_envs: # of parallel environments to simulation
        :param use_omega: use the omega controller, which converts desired (thrust, omega) to motor forces
        :param action_is_mf: specified that the action space is motor forces
        :param convert_mf_to_omega: converts a motor force command to desired (thrust, omega) to model Crazyflie
        :param use_delay_model: use the delay model to translate desired (thrust, omega) into actual thrust and omega
        :param delay_coeff: coefficient of delay model
        :param tensor_args: PyTorch tensor arguments
        """
        super().__init__()
        self.num_envs = num_envs
        self.use_omega = use_omega
        self.action_is_mf = action_is_mf
        self.convert_mf_to_omega = convert_mf_to_omega
        self.use_delay_model = use_delay_model
        self.delay_coeff = delay_coeff
        self.tensor_args = tensor_args

        # Get the quadrotor configuration
        self.config = yaml.load(open(config), yaml.FullLoader)
        self.param = QuadrotorParam(self.config, MPPI=True)

        # Init timing
        self.times = self.param.sim_times
        self.time_step = 0
        self.avg_dt = self.times[1] - self.times[0]

        # Init system state
        self.init_state = torch.tensor(self.config['initial_state'], **self.tensor_args)

        # Control bounds
        self.a_min = torch.tensor(self.param.a_min, **self.tensor_args)
        self.a_max = torch.tensor(self.param.a_max, **self.tensor_args)

        if (not self.action_is_mf and not self.convert_mf_to_omega) or use_omega:
            self.action_lows = torch.tensor([0., -10, -10, -10], **self.tensor_args)
            #self.action_highs = torch.tensor([self.a_max[0]*4, 12, 12, 12], **self.tensor_args)
            self.action_highs = torch.tensor([0.7848, 10, 10, 10], **self.tensor_args)
        else:
            self.action_lows = self.a_min
            self.action_highs = self.a_max

        # Initial conditions
        self.s_min = torch.tensor(self.param.s_min, **self.tensor_args)
        self.s_max = -self.s_min
        self.rpy_limit = torch.tensor(self.param.rpy_limit, **self.tensor_args)
        self.limits = torch.tensor(self.param.limits, **self.tensor_args)

        # Constants
        self.d_state = 13
        self.d_obs = 13
        self.d_action = 4

        self.mass = self.param.mass
        self.g = self.param.g
        self.inv_mass = 1 / self.mass

        self.d = self.param.d
        self.rho = self.param.rho
        self.Cs = self.param.Cs
        self.Ct = self.param.Ct
        self.k1 = self.param.k1
        self.k2 = self.param.k2

        self.B0 = torch.tensor(self.param.B0, **self.tensor_args)
        self.B0_inv = torch.linalg.inv(self.B0)

        self.J = torch.tensor(self.param.J, **self.tensor_args)
        if self.J.shape == (3, 3):
            self.J = torch.as_tensor(self.J, **self.tensor_args)
            self.inv_J = torch.linalg.inv(self.J)
        else:
            self.J = torch.diag(torch.as_tensor(self.J, **self.tensor_args))
            self.inv_J = torch.linalg.inv(self.J)

        # Controller gains
        self.omega_gain = self.config['omega_gain']

        # Plotting stuff
        self.states_name = [
            'Position X [m]',
            'Position Y [m]',
            'Position Z [m]',
            'Velocity X [m/s]',
            'Velocity Y [m/s]',
            'Velocity Z [m/s]',
            'qw',
            'qx',
            'qy',
            'qz',
            'Angular Velocity X [rad/s]',
            'Angular Velocity Y [rad/s]',
            'Angular Velocity Z [rad/s]']

        self.deduced_state_names = [
            'Roll [deg]',
            'Pitch [deg]',
            'Yaw [deg]',
        ]

        self.actions_name = [
            'Motor Force 1 [N]',
            'Motor Force 2 [N]',
            'Motor Force 3 [N]',
            'Motor Force 4 [N]']

    def f(self, s: Tensor, a: Tensor) -> Tensor:
        num_envs, num_samples, d_state = s.shape
        dsdt = torch.zeros(num_envs*num_samples, 13).to(**self.tensor_args)
        v = s[:, :, 3:6].view(-1, 3)  # velocity (N, 3)
        q = s[:, :, 6:10].view(-1, 4)  # quaternion (N, 4)
        omega = s[:, :, 10:].view(-1, 3)  # angular velocity (N, 3)
        a = a.view(-1, 4)

        if self.action_is_mf and not self.convert_mf_to_omega:
            # If action space is motor forces and we did not convert to omega space, then compute wrench
            eta = a @ self.B0.T # output wrench (N, 4)
        else:
            # Otherwise, our action is (thrust, omega)
            eta = a

        f_u = torch.zeros(num_envs*num_samples, 3).to(**self.tensor_args)
        f_u[:, 2] = eta[:, 0]  # total thrust (N, 3)
        tau_u = eta[:, 1:]  # torque (N, 3)

        # dynamics
        # \dot{p} = v
        dsdt[:, :3] = v  # <- implies velocity and position in same frame

        # mv = mg + R f_u  	# <- implies f_u in body frame, p, v in world frame
        dsdt[:, 5] -= self.g
        dsdt[:, 3:6] += qrotate_torch(q, f_u) / self.mass

        # \dot{R} = R S(w)
        # see https://rowan.readthedocs.io/en/latest/package-calculus.html
        qnew = qintegrate_torch(q, omega, self.avg_dt, frame='body')
        qnew = qstandardize_torch(qnew)

        # transform qnew to a "delta q" that works with the usual euler integration
        dsdt[:, 6:10] = (qnew - q) / self.avg_dt

        if self.action_is_mf and not self.convert_mf_to_omega:
            # Compute omega from torques
            # J\dot{w} = Jw x w + tau_u
            Jomega = omega @ self.J.T
            dsdt[:, 10:] = torch.cross(Jomega, omega) + tau_u
            dsdt[:, 10:] = dsdt[:, 10:] @ self.inv_J.T
        else:
            # Set updated omega to be the control command
            dsdt[:, 10:] = (tau_u - omega) / self.avg_dt

        dsdt = dsdt.view(num_envs, num_samples, -1)
        return dsdt

    def step(self, s: Tensor, a: Tensor) -> Tensor:
        new_s = s + self.avg_dt * self.f(s, a)
        return new_s

    def rollout_open_loop(self, start_state: Tensor, act_seq: Tensor) -> Dict[str, Any]:
        num_particles, horizon, _ = act_seq.shape
        state_seq = torch.zeros((num_particles, horizon, self.d_state), **self.tensor_args)
        state_t = start_state.repeat((num_particles, 1))

        for t in range(horizon):
           state_t = self.step(state_t, act_seq[:,t])
           state_seq[:, t] = state_t

        trajectories = dict(
            actions = act_seq,
            state_seq = state_seq
        )

        return trajectories
    
    def get_next_state(self, curr_state, act, dt):
        pass