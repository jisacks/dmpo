import numpy as np
import torch

from torch import Tensor
from numpy.typing import NDArray
from typing import Dict, Any, Optional, List, Union, Tuple

class QuadrotorParam():
	"""
	Class containing parameters for quadrotor model
	"""
	def __init__(self,
				 config: Dict[str, Any],
				 MPPI: bool=False):
		"""
		:param config: dictionary containing configuration
		:param MPPI: indicates if we should use the MPPI controller parameters (which may be different)
		"""
		self.env_name = 'Quadrotor-Crazyflie'

		# Quadrotor parameters
		self.sim_t0 = 0
		self.sim_tf = config['sim_tf']
		self.sim_dt = config['sim_dt']
		self.sim_times = np.arange(self.sim_t0, self.sim_tf, self.sim_dt)
		if MPPI:
			self.sim_dt = config['sim_dt_MPPI']
			self.sim_times = np.arange(self.sim_t0, self.sim_tf, self.sim_dt)

		# Control limits [N] for motor forces
		self.a_min = np.array(config.get('a_min', [0., 0, 0, 0]))
		self.a_max = np.array(config.get('a_max', [12., 12., 12., 12])) / 1000 * 9.81 # g->N

		# Crazyflie 2.0 quadrotor.py
		self.mass = config.get('mass', 0.034) #kg
		self.J = np.array(config.get('J', [16.571710, 16.655602, 29.261652])) * 1e-6
		self.d = 0.047

		# Side force model parameters for wind perturbations
		if config['Vwind'] == 0:
			self.wind = False
			self.Vwind = None
		else:
			self.wind = True
			self.Vwind = np.array(config['Vwind']) # velocity of wind in world frame
		self.Ct = 2.87e-3
		self.Cs = 2.31e-5
		self.k1 = 1.425
		self.k2 = 3.126
		self.rho = 1.225 # air density (in SI units)

		# Note: we assume here that our control is forces
		arm_length = 0.046 # m
		arm = 0.707106781 * arm_length
		t2t = 0.006 # thrust-to-torque ratio
		self.t2t = t2t
		self.B0 = np.array([
			[1, 1, 1, 1],
			[-arm, -arm, arm, arm],
			[-arm, arm, arm, -arm],
			[-t2t, t2t, -t2t, t2t]
			])
		self.g = 9.81 # not signed

		# Exploration parameters: state boundary and initial state sampling range
		self.s_min = np.array( \
			[-8, -8, -8, \
			  -5, -5, -5, \
			  -1.001, -1.001, -1.001, -1.001,
			  -20, -20, -20])
		self.rpy_limit = np.array([5, 5, 5])
		self.limits = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0])

		# Measurement noise
		self.noise_measurement_std = np.zeros(13)
		self.noise_measurement_std[:3] = 0.005
		self.noise_measurement_std[3:6] = 0.005
		self.noise_measurement_std[6:10] = 0.01
		self.noise_measurement_std[10:] = 0.01

		# Process noise
		self.noise_process_std = config.get('noise_process_std', [0.3, 2.])

		# Reference trajectory parameters
		self.ref_type = config['traj']
		self.max_dist = config.get('max_dist', [1., 1., 0.])
		self.min_dt = config.get('min_dt', 0.6)
		self.max_dt = config.get('max_dt', 1.5)

		# Cost function parameters
		self.alpha_p = config['alpha_p']
		self.alpha_z = config['alpha_z']
		self.alpha_w = config['alpha_w']
		self.alpha_a = config['alpha_a']
		self.alpha_R = config['alpha_R']
		self.alpha_v = config['alpha_v']
		self.alpha_yaw = config['alpha_yaw']
		self.alpha_pitch = config['alpha_pitch']
		self.alpha_u_delta = config['alpha_u_delta']
		self.alpha_u_thrust = config['alpha_u_thrust']
		self.alpha_u_omega = config['alpha_u_omega']

	def get_reference(self,
					  num_envs: int,
					  dts: Optional[NDArray]=None,
					  pos: Optional[NDArray]=None) -> Tuple[Tensor, NDArray, NDArray]:
		"""
		:param num_envs: # of reference trajectories to generate (one per environment)
		:param dts: delta in time between waypoints (randomly generated if not specified)
		:param pos: waypoints for zig-zag (randomly generated if not specified)
		"""

		self.ref_trajectory = np.zeros((num_envs, 13, len(self.sim_times)))
		self.ref_trajectory[:, 6, :] = 1.

		if self.ref_type == 'zig-zag-yaw':
			if dts is None:
				dts = np.random.uniform(self.min_dt, self.max_dt, size=(num_envs, len(self.sim_times), 1))
				dts = dts.repeat(3, axis=2)
			d_times = dts.cumsum(axis=1)
			if pos is None:
				pos = np.random.uniform(0, np.array(self.max_dist), size=(num_envs, len(self.sim_times)//2, 3))

			for env_idx in range(num_envs):
				for p_idx in range(3):
					for step, time in enumerate(self.sim_times[1:]):
						ref_idx = np.searchsorted(d_times[env_idx, :, p_idx], time)
						sign = 1 if np.ceil(ref_idx/2 - ref_idx//2) == 0 else -1
						ref_pos = sign * pos[env_idx, ref_idx, p_idx]
						prev_ref_pos = -1 * sign * pos[env_idx, ref_idx-1, p_idx] if ref_idx > 0 else 0
						ref_time = d_times[env_idx, ref_idx, p_idx]
						prev_ref_time = d_times[env_idx, ref_idx-1, p_idx] if ref_idx > 0 else 0
						cur_pos = self.ref_trajectory[env_idx, p_idx, step]

						delta = ref_pos - prev_ref_pos
						if delta != 0:
							delta = delta/(ref_time-prev_ref_time)*self.sim_dt
						self.ref_trajectory[env_idx, p_idx, step+1] = cur_pos + delta
						self.ref_trajectory[env_idx, 6, step+1] = 1 if sign == 1 else 0
						self.ref_trajectory[env_idx, 9, step+1] = 1 if sign == -1 else 0
		elif self.ref_type == 'zig-zag':
			if dts is None:
				dts = np.random.uniform(self.min_dt, self.max_dt, size=(num_envs, len(self.sim_times), 1))
				dts = dts.repeat(3, axis=2)
			d_times = dts.cumsum(axis=1)
			if pos is None:
				pos = np.random.uniform(0, np.array(self.max_dist), size=(num_envs, len(self.sim_times)//2, 3))

			for env_idx in range(num_envs):
				for p_idx in range(3):
					for step, time in enumerate(self.sim_times[1:]):
						ref_idx = np.searchsorted(d_times[env_idx, :, p_idx], time)
						sign = 1 if np.ceil(ref_idx/2 - ref_idx//2) == 0 else -1
						ref_pos = sign * pos[env_idx, ref_idx, p_idx]
						prev_ref_pos = -1 * sign * pos[env_idx, ref_idx-1, p_idx] if ref_idx > 0 else 0
						ref_time = d_times[env_idx, ref_idx, p_idx]
						prev_ref_time = d_times[env_idx, ref_idx-1, p_idx] if ref_idx > 0 else 0
						cur_pos = self.ref_trajectory[env_idx, p_idx, step]

						delta = ref_pos - prev_ref_pos
						if delta != 0:
							delta = delta/(ref_time-prev_ref_time)*self.sim_dt
						self.ref_trajectory[env_idx, p_idx, step+1] = cur_pos + delta
		else:
			raise ValueError('Invalid reference trajectory type specified.')

		return torch.tensor(self.ref_trajectory), dts, pos


