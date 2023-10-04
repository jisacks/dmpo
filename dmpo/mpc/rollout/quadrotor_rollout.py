from .rollout_base import RolloutBase
from ..model.quadrotor_model import QuadrotorModel
from ...envs.math_utils import *

import torch

from torch import Tensor
from typing import Dict, Any, Optional, List

class QuadrotorRollout(RolloutBase):
    def __init__(self,
                 dynamics_model: QuadrotorModel,
                 exp_params: Dict[str, Any],
                 num_envs: int = 1,
                 tensor_args: Dict[str, Any] = {'device': 'cpu', 'dtype': torch.float32}):
        """
        :param dynamics_model: Dynamics model used in rollout
        :param exp_params: Cost function parameters
        :param num_envs: Number of environments
        :param tensor_args: PyTorch params
        """
        super(QuadrotorRollout, self).__init__()
        self.dynamics_model = dynamics_model
        self.tensor_args = tensor_args
        self.exp_params = exp_params
        self.num_envs = num_envs

        self.use_omega = dynamics_model.use_omega
        self.action_is_mf = dynamics_model.action_is_mf
        self.convert_mf_to_omega = dynamics_model.convert_mf_to_omega
        self.use_delay_model = dynamics_model.use_delay_model

        self.param = self.dynamics_model.param
        self.alpha_p = self.param.alpha_p
        self.alpha_z = self.param.alpha_z
        self.alpha_w = self.param.alpha_w
        self.alpha_a = self.param.alpha_a
        self.alpha_R = self.param.alpha_R
        self.alpha_v = self.param.alpha_v
        self.alpha_pitch = self.param.alpha_pitch
        self.alpha_yaw = self.param.alpha_yaw
        self.t = torch.zeros((self.num_envs), **self.tensor_args)

    def cost_fn(self, state: Tensor, act: Tensor) -> Dict[str, Tensor]:
        time_step = torch.ceil(self.t / self.dynamics_model.avg_dt)
        num_envs, num_samples, H, _ = state.shape
        indices = torch.arange(H).to(**self.tensor_args) + time_step[0]
        indices = indices.clip(0, self.ref_trajectory.shape[-1]-1)
        indices = indices.to(torch.long)

        state_ref = self.ref_trajectory[:, :, indices].permute((0, 2, 1))
        p_des = state_ref[:, :, 0:3]
        v_des = state_ref[:, :, 3:6]
        w_des = state_ref[:, :, 10:]
        q_des = state_ref[:, :, 6:10]

        # Position tracking error
        if self.alpha_p > 0:
            ep = torch.linalg.norm(state[:, :, :, 0:3] - p_des[:, None], dim=-1)
        else:
            ep = 0.

        # Additional cost on Z tracking error
        if self.alpha_z > 0:
            ez = torch.linalg.norm(state[:, :, :, 2:3] - p_des[:, None, :, 2:3], dim=-1)
        else:
            ez = 0.

        # Velocity tracking error
        if self.alpha_v > 0:
            ev = torch.linalg.norm(state[:, :, :, 3:6] - v_des[:, None], dim=-1)
        else:
            ev = 0.

        # Angular velocity tracking error
        if self.alpha_w > 0:
            ew = torch.linalg.norm(state[:, :, :, 10:] - w_des[:, None], dim=-1)
        else:
            ew = 0.

        # Orientation tracking error
        if self.alpha_R > 0:
            q_des_repeated = q_des[:, None].repeat(1, num_samples, 1, 1)
            eR = qdistance_torch(state[:, :, :, 6:10].view(-1, 4), q_des_repeated.view(-1, 4))
            eR = eR.view(num_envs, num_samples, H)
        else:
            eR = 0.

        # Control cost
        if self.alpha_a > 0:
            ea = torch.linalg.norm(act, dim=-1)
        else:
            ea = 0.

        # Yaw tracking error
        if self.alpha_yaw > 0:
            q_des_repeated = q_des[:, None, :, :].repeat(1, num_samples, 1, 1).view(-1, 4)
            qe = qmultiply_torch(qconjugate_torch(q_des_repeated), state[:, :, :, 6:10].view(-1, 4))
            Re = qtoR_torch(qe)
            eyaw = torch.atan2(Re[:, 1, 0], Re[:, 0, 0]) ** 2
            eyaw = eyaw.view(num_envs, num_samples, H)
        else:
            eyaw = 0.

        # Pitch tracking error
        if self.alpha_pitch > 0:
            q_des_repeated = q_des[:, None, :, :].repeat(1, num_samples, 1, 1).view(-1, 4)
            qe = qmultiply_torch(qconjugate_torch(q_des_repeated), state[:, :, :, 6:10].view(-1, 4))
            Re = qtoR_torch(qe)
            epitch = (torch.asin(Re[:,2,0].clip(-1, 1)))**2
            epitch = epitch.view(num_envs, num_samples, H)
        else:
            epitch = 0.

        cost = (self.alpha_p * ep
                + self.alpha_z * ez
                + self.alpha_v * ev
                + self.alpha_w * ew
                + self.alpha_a * ea
                + self.alpha_yaw * eyaw
                + self.alpha_R * eR
                + self.alpha_pitch * epitch) * self.dynamics_model.avg_dt

        return dict(
            cost=cost,
            ep=self.alpha_p * ep * self.dynamics_model.avg_dt,
            ez=self.alpha_z * ez * self.dynamics_model.avg_dt,
            ev=self.alpha_v * ev * self.dynamics_model.avg_dt,
            ew=self.alpha_w * ew * self.dynamics_model.avg_dt,
            ea=self.alpha_a * ea * self.dynamics_model.avg_dt,
            eyaw=self.alpha_yaw * eyaw * self.dynamics_model.avg_dt,
            eR=self.alpha_R * eR * self.dynamics_model.avg_dt,
            epitch=self.alpha_pitch * epitch * self.dynamics_model.avg_dt
        )

    def omega_controller(self, s: Tensor, a: Tensor) -> Tensor:
        # Converts desired (thrust, omega) to motor forces
        num_envs, num_samples, _ = a.shape

        T_d = a[:, :, 0].reshape(-1)
        omega_d = a[:, :, 1:].reshape(-1, 3)
        omega = s[:, :, 10:13].reshape(-1, 3)
        omega_e = omega_d - omega

        torque = self.dynamics_model.omega_gain * omega_e  # tensor, (3,)
        torque = torch.mm(self.dynamics_model.J, torque.T).T
        torque -= torch.cross(torch.mm(self.dynamics_model.J, omega.T).T, omega)

        wrench = torch.cat((T_d.view(-1, 1), torque), dim=1)  # tensor, (N, 4)
        motorForce = torch.mm(self.dynamics_model.B0_inv, wrench.T).T
        motorForce = torch.clip(motorForce, self.dynamics_model.a_min, self.dynamics_model.a_max)
        motorForce = motorForce.view(num_envs, num_samples, -1)
        return motorForce

    def convert_motor_forces(self, s: Tensor, a: Tensor) -> Tensor:
        '''
        Converts motor forces to desired (thrust, omega)
        '''
        num_envs, num_particles, _ = a.shape
        eta = a.view(num_envs*num_particles, -1) @ self.dynamics_model.B0.T
        T_d = eta[:, :1]
        tau_u = eta[:, 1:]

        omega = s[:, :, 10:].view(num_envs*num_particles, -1)
        Jomega = omega @ self.dynamics_model.J.T
        d_omega = torch.cross(Jomega, omega) + tau_u
        d_omega = d_omega @ self.dynamics_model.inv_J.T
        omega_d = omega + d_omega*self.dynamics_model.avg_dt

        new_a = torch.cat((T_d, omega_d), dim=-1)
        new_a = new_a.view(num_envs, num_particles, -1)
        return new_a

    def rollout_fn(self, start_state: Tensor, act_seq: Tensor) -> Dict[str, Any]:
        actions = act_seq
        if actions.ndim == 3:
            actions = actions.unsqueeze(0)

        num_envs, num_particles, horizon, _ = actions.shape
        state_seq = torch.zeros((num_envs, num_particles, horizon, start_state.shape[-1]), **self.tensor_args)
        states = start_state.unsqueeze(1).repeat((1, num_particles, 1))

        # Apply delay model if action space is not motor forces
        if not self.action_is_mf and self.use_delay_model:
            if self.actions is None:
                self.actions = self.env_actions.unsqueeze(2).repeat(1, num_particles, horizon, 1)
            action = torch.clip(actions, self.dynamics_model.action_lows, self.dynamics_model.action_highs)
            self.actions = self.actions + self.dynamics_model.delay_coeff * (actions - self.actions)

        for t in range(horizon):
            if not self.action_is_mf and not self.convert_mf_to_omega:
                # Ensure control bounds if using desired (thrust, omega) as control space
                action = actions[:, :, t]
                action = torch.clip(action, self.dynamics_model.action_lows, self.dynamics_model.action_highs)
            elif self.convert_mf_to_omega:
                # Convert motor forces to desired (thrust, omega) command
                action = actions[:, :, t]
                action = torch.clip(action, self.dynamics_model.a_min, self.dynamics_model.a_max)
                action = self.convert_motor_forces(states, action)
            elif self.action_is_mf:
                # Ensure control bounds if using motor forces
                action = actions[:, :, t]
                action = torch.clip(action, self.dynamics_model.a_min, self.dynamics_model.a_max)

            if self.use_omega:
                # Apply omega controller to convert motor forces to desired (thrust, omega)
                action = self.omega_controller(states, actions[:, :, t])
            elif self.use_delay_model:
                # Get action from the delayed actions
                action = self.actions[:, :, t]

            states = self.dynamics_model.step(states, action)
            state_seq[:, :, t] = states

        cost_seq = self.cost_fn(state_seq, act_seq)

        trajectories = dict(
            actions=act_seq,
            costs=cost_seq,
            rollout_time=0.0,
            state_seq=state_seq
        )

        return trajectories

    def __call__(self, start_state: Tensor, act_seq: Tensor) -> Dict[str, Any]:
        return self.rollout_fn(start_state, act_seq)

    def update_params(self, t, actions, ref_dts, ref_pos):
        self.t = torch.tensor(t, **self.tensor_args)

        # Reset stored actions if at initial time point
        self.env_actions = torch.stack(actions).to(**self.tensor_args)[:, None]
        if np.any(t) == 0:
           self.actions = None

        # Get the reference trajectory
        kwargs = {}
        if not ref_dts is None:
            kwargs['dts'] = np.stack(ref_dts)
        if not ref_pos is None:
            kwargs['pos'] = np.stack(ref_pos)

        self.num_envs = len(ref_pos)
        self.ref_trajectory, _, _ = self.param.get_reference(self.num_envs, **kwargs)
        self.ref_trajectory = self.ref_trajectory.to(**self.tensor_args)
        return True