import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import rowan
import time
from copy import deepcopy
import yaml
from math import ceil

from .quadrotor_param import QuadrotorParam
from .math_utils import *

from torch import Tensor
from typing import Dict, Any, Optional, List, Tuple

class QuadrotorEnv():
    """
    Quadrotor simulation environment
    """
    def __init__(self,
                 config: str,
                 num_envs: int=1,
                 use_omega: bool=False,
                 action_is_mf: bool=True,
                 convert_mf_to_omega: bool=False,
                 use_delay_model: bool=False,
                 delay_coeff: float=0.2,
                 randomize_mass: bool = False,
                 mass_range: List[float]=[0.7, 1.3],
                 randomize_delay_coeff: bool = False,
                 delay_range: List[float]=[0.2, 0.6],
                 force_pert: bool=False,
                 force_range: List[float]=[-3.5, 3.5],
                 force_is_z: bool=False,
                 ou_theta: float=0.15,
                 ou_sigma: float=0.20,
                 use_obs_noise: bool=False,
                 tensor_args: Dict[str, Any]={'device': 'cpu', 'dtype': torch.float32}):
        """
        :param config: YAML configuration file, which will be parsed to form a QuadrotorParam object
        :param num_envs: # of parallel environments to simulation
        :param use_omega: use the omega controller, which converts desired (thrust, omega) to motor forces
        :param action_is_mf: specified that the action space is motor forces
        :param convert_mf_to_omega: converts a motor force command to desired (thrust, omega) to model Crazyflie
        :param use_delay_model: use the delay model to translate desired (thrust, omega) into actual thrust and omega
        :param delay_coeff: coefficient of delay model
        :param randomize_mass: use domain randomization for mass
        :param mass_range: scaling factors for mass for domain randomization
        :param randomize_delay_coeff: use domain randomization for delay coefficient
        :param delay_range: range of delay coefficients to use for domain randomization
        :param force_pert: use random force perturbations
        :param force_range: range of force perturbations
        :param force_is_z: force perturbations can also be in Z direction (will just be XY if false)
        :param ou_theta: OU process theta parameter for changing force perturbation over time
        :param ou_sigma: OU process sigma parameter for changing force perturbation over time
        :param use_obs_noise: corrupt state with observation noise
        :param tensor_args: PyTorch tensor arguments
        """
        self.num_envs = num_envs
        self.tensor_args = tensor_args

        # Load in the configuration
        self.config = yaml.load(open(config), yaml.FullLoader)
        self.param = QuadrotorParam(self.config)

        # Action space parameters
        self.use_omega = use_omega
        self.action_is_mf = action_is_mf
        self.convert_mf_to_omega = convert_mf_to_omega

        # Delay model parameters
        self.use_delay_model = use_delay_model
        self.true_delay_coeff = delay_coeff

        # Domain randomization parameters
        self.mass_range = mass_range
        self.randomize_mass = randomize_mass
        self.delay_range = delay_range
        self.randomize_delay_coeff = randomize_delay_coeff
        self.force_pert = force_pert
        self.force_range = force_range
        self.force_is_z = force_is_z
        self.use_obs_noise = use_obs_noise
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma

        # Init timing
        self.times = self.param.sim_times
        self.time_step = 0
        self.avg_dt = self.times[1] - self.times[0]

        # Init system state
        self.init_state = torch.tensor(self.config['initial_state'], **self.tensor_args)

        # Control bounds in motor force space
        self.a_min = torch.tensor(self.param.a_min, **self.tensor_args)
        self.a_max = torch.tensor(self.param.a_max, **self.tensor_args)

        # Get action bounds for controller (may be motor force space if specified)
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

        self.mass = torch.tensor([self.param.mass], **self.tensor_args).repeat(self.num_envs)
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

        # Reward function coefficients
        # ref: row 8, Table 3, USC sim-to-real paper
        self.alpha_p = self.param.alpha_p
        self.alpha_z = self.param.alpha_z
        self.alpha_w = self.param.alpha_w
        self.alpha_a = self.param.alpha_a
        self.alpha_R = self.param.alpha_R
        self.alpha_v = self.param.alpha_v
        self.alpha_yaw = self.param.alpha_yaw
        self.alpha_pitch = self.param.alpha_pitch
        self.alpha_u_delta = self.param.alpha_u_delta
        self.alpha_u_thrust = self.param.alpha_u_thrust
        self.alpha_u_omega = self.param.alpha_u_omega

    def get_env_state(self) -> List[Tensor]:
        return self.states

    def set_env_state(self, state: List[Tensor]) -> None:
        self.states = state

    def set_param(self, param_dict: Dict[str, Any]) -> None:
        for k, v in param_dict.items():
            setattr(self, k, v)

    def reset(self) -> Tensor:
        # Determine the initial state of the quadrotor
        if self.init_state is None:
            self.states = torch.zeros(self.d_state)

            # Position and velocity
            limits = self.limits
            self.states[0:6] = torch.rand(6) * (2 * limits[0:6]) - limits[0:6]

            # Rotation
            rpy = np.radians(np.random.uniform(-self.rpy_limit, self.rpy_limit, 3))
            q = rowan.from_euler(rpy[0], rpy[1], rpy[2], 'xyz')
            self.states[6:10] = torch.tensor(q, **self.tensor_args)

            # Angular velocity
            self.states[10:13] = torch.rand(3) * (2 * limits[10:13]) - limits[10:13]
        else:
            self.states = self.init_state

        # Reset state and action variables
        self.states = self.states.unsqueeze(0).repeat(self.num_envs, 1)
        self.time_step = 0
        self.actions = torch.zeros((self.num_envs, self.d_action), **self.tensor_args)
        self.prev_actions = torch.zeros((self.num_envs, self.d_action), **self.tensor_args)

        # Randomize mass
        if self.randomize_mass:
            mass_scale = torch.rand(self.num_envs, **self.tensor_args)*(self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
            self.mass = torch.tensor([self.param.mass], **self.tensor_args).repeat(self.num_envs)
            self.mass = self.mass * mass_scale
            self.inv_mass = 1 / self.mass
        else:
            self.mass = torch.tensor([self.param.mass], **self.tensor_args).repeat(self.num_envs)
            self.inv_mass = 1 / self.mass

        # Randomize delay coeff:
        if self.randomize_delay_coeff:
            self.delay_coeff = torch.rand(self.num_envs, **self.tensor_args) * (self.delay_range[1] - self.delay_range[0]) + self.delay_range[0]
        else:
            self.delay_coeff = torch.tensor([self.true_delay_coeff], **self.tensor_args).repeat(self.num_envs)

        if self.force_pert:
            self.force_dist = torch.rand((self.num_envs, 3), **self.tensor_args) * (self.force_range[1] - self.force_range[0]) + self.force_range[0]
            if not self.force_is_z:
                self.force_dist[:, -1] = 0
        else:
            self.force_dist = torch.zeros((self.num_envs, 3), **self.tensor_args)

        # Get the reference trajectories
        self.ref_trajectory, self.ref_dts, self.ref_pos = self.param.get_reference(self.num_envs)
        self.ref_trajectory = self.ref_trajectory.to(**self.tensor_args)

        return self.states

    def get_env_obs(self) -> List[Tensor]:
        if self.use_obs_noise:
            noise = torch.randn((self.num_envs, self.d_state), **self.tensor_args)
            noise = noise * torch.as_tensor(self.param.noise_measurement_std, **self.tensor_args)
            noisystate = self.states + noise
            noisystate[:, 6:10] /= torch.norm(noisystate[:, 6:10], dim=-1, keepdim=True)
            return noisystate
        else:
            return self.states

    def f(self, s: Tensor, a: Tensor) -> Tensor:
        num_envs = s.shape[0]
        dsdt = torch.zeros(num_envs, 13).to(**self.tensor_args)
        v = s[:, 3:6]  # velocity (N, 3)
        q = s[:, 6:10]  # quaternion (N, 4)
        omega = s[:, 10:]  # angular velocity (N, 3)

        if self.action_is_mf and not self.convert_mf_to_omega:
            # If action space is motor forces and we did not convert to omega space, then compute wrench
            eta = a @ self.B0.T # output wrench (N, 4)
        else:
            # Otherwise, our action is (thrust, omega)
            eta = a

        f_u = torch.zeros(num_envs, 3).to(**self.tensor_args)
        f_u[:, 2] = eta[:, 0]  # total thrust (N, 3)
        tau_u = eta[:, 1:]  # torque (N, 3) or desired omega

        # dynamics
        # \dot{p} = v
        dsdt[:, :3] = v  # <- implies velocity and position in same frame

        # Apply the force perturbation
        if self.force_pert:
            dsdt[:, 3:6] += self.force_dist

        # mv = mg + R f_u  	# <- implies f_u in body frame, p, v in world frame
        dsdt[:, 5] -= self.g
        dsdt[:, 3:6] += qrotate_torch(q, f_u) / self.mass[:, None]

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

        # Adding noise
        dsdt[:, 3:6] += torch.normal(mean=0,
                                     std=self.param.noise_process_std[0],
                                     size=(self.num_envs, 3),
                                     **self.tensor_args)
        dsdt[:, 10:] += torch.normal(mean=0,
                                     std=self.param.noise_process_std[1],
                                     size=(self.num_envs, 3),
                                     **self.tensor_args)

        return dsdt

    def next_state(self, s: Tensor, a: Tensor) -> Tensor:
        new_s = s + self.avg_dt * self.f(s, a)
        return new_s

    def get_cost(self, a: Tensor) -> Tensor:
        state_ref = self.ref_trajectory[:, :, self.time_step]
        p_des = state_ref[:, 0:3]
        v_des = state_ref[:, 3:6]
        w_des = state_ref[:, 10:]
        q_des = state_ref[:, 6:10]

        # Position tracking error
        if self.alpha_p > 0:
            ep = torch.linalg.norm(self.states[:, 0:3] - p_des, dim=1)
        else:
            ep = 0.

        # Additional cost on Z tracking error
        if self.alpha_z > 0:
            ez = torch.linalg.norm(self.states[:, 2:3] - p_des[2:3], dim=-1)
        else:
            ez = 0.

        # Velocity tracking error
        if self.alpha_v > 0:
            ev = torch.linalg.norm(self.states[:, 3:6] - v_des, dim=1)
        else:
            ev = 0.

        # Angular velocity tracking error
        if self.alpha_w > 0:
            ew = torch.linalg.norm(self.states[:, 10:] - w_des, dim=1)
        else:
            ew = 0.

        # Orientation tracking error
        if self.alpha_R > 0:
            eR = qdistance_torch(self.states[:, 6:10], q_des)
        else:
            eR = 0.

        # Control cost
        if self.alpha_a > 0:
            ea = torch.linalg.norm(a, dim=1)
        else:
            ea = 0.

        # Yaw tracking error
        if self.alpha_yaw > 0:
            qe = qmultiply_torch(qconjugate_torch(q_des), self.states[:, 6:10])
            Re = qtoR_torch(qe)
            eyaw = torch.atan2(Re[:, 1, 0], Re[:, 0, 0]) ** 2
        else:
            eyaw = 0.

        # Pitch tracking error
        if self.alpha_pitch > 0:
            qe = qmultiply_torch(qconjugate_torch(q_des), self.states[:, 6:10])
            Re = qtoR_torch(qe)
            epitch = (torch.asin(Re[:,2,0].clip(-1, 1)))**2
        else:
            epitch = 0

        # Penalize control changes between time steps
        if self.alpha_u_delta > 0:
            edelta = torch.norm(a - self.prev_actions, dim=1)
        else:
            edelta = 0

        # Separate thrust control cost (use only if control space includes thrust)
        if self.alpha_u_thrust > 0:
            ethrust = torch.norm(a[:, :1], dim=1)
        else:
            ethrust = 0

        # Separate omega control cost (use only if control space includes omega)
        if self.alpha_u_omega > 0:
            eomega = torch.norm(a[:, 1:], dim=1)
        else:
            eomega = 0

        cost = (self.alpha_p * ep
                + self.alpha_z * ez
                + self.alpha_v * ev
                + self.alpha_w * ew
                + self.alpha_a * ea
                + self.alpha_yaw * eyaw
                + self.alpha_R * eR
                + self.alpha_pitch * epitch
                + self.alpha_u_delta * edelta
                + self.alpha_u_thrust * ethrust
                + self.alpha_u_omega * eomega) * self.avg_dt
        return cost

    def get_env_infos(self) -> Dict[str, Any]:
        done = (self.time_step + 1) >= len(self.times)
        dones = [done for _ in range(self.num_envs)]
        return dict(dones=dones, done=done)

    def omega_controller(self, a: Tensor) -> Tensor:
        '''
        Converts desired (thrust, omega) to motor forces
        '''
        T_d = a[:, 0]
        omega_d = a[:, 1:]

        omega = self.states[:, 10:13]
        omega_e = omega_d - omega

        torque = self.omega_gain * omega_e  # tensor, (3,)
        torque = torch.mm(self.J, torque.T).T
        torque -= torch.cross(torch.mm(self.J, omega.T).T, omega)

        wrench = torch.cat((T_d.view(self.num_envs, 1), torque), dim=1)  # tensor, (N, 4)
        motorForce = torch.mm(self.B0_inv, wrench.T).T
        motorForce = torch.clip(motorForce, self.a_min, self.a_max)
        return motorForce

    def convert_motor_forces(self, a: Tensor) -> Tensor:
        '''
        Converts motor forces to desired (thrust, omega)
        '''
        eta = a @ self.B0.T
        T_d = eta[:, :1]
        tau_u = eta[:, 1:]

        omega = self.states[:, 10:]
        Jomega = omega @ self.J.T
        d_omega = torch.cross(Jomega, omega) + tau_u
        d_omega = d_omega @ self.inv_J.T
        omega_d = omega + d_omega*self.avg_dt

        new_a = torch.cat((T_d, omega_d), dim=-1)
        return new_a

    def step(self, a: Tensor) -> Tuple[Any, ...]:
        a = a.to(**self.tensor_args)

        if not self.action_is_mf and not self.convert_mf_to_omega:
            # Ensure control bounds if using desired (thrust, omega) as control space
            a = torch.clip(a, self.action_lows, self.action_highs)
        elif self.convert_mf_to_omega:
            # Convert motor forces to desired (thrust, omega) command
            a = torch.clip(a, self.a_min, self.a_max)
            a = self.convert_motor_forces(a)
        elif self.action_is_mf:
            # Ensure control bounds if using motor forces
            a = torch.clip(a, self.a_min, self.a_max)

        # Apply omega controller to convert motor forces to desired (thrust, omega)
        if self.use_omega:
            a = self.omega_controller(a)
        cmd = a.clone()

        # Apply delay model on controls (should only be used for desired thrust, omega action space)
        if self.use_delay_model:
            self.actions = self.actions + self.delay_coeff[:, None]*(a - self.actions)
            a = self.actions

        # Compute the state transitions
        new_states = self.next_state(self.states, a)
        self.states = new_states

        # Increment force perturbation with OU process
        if self.force_pert:
            d_force = -self.ou_theta * self.force_dist
            d_force += torch.randn(d_force.shape, **self.tensor_args) * self.ou_sigma
            self.force_dist += d_force * self.avg_dt
            self.force_dist = self.force_dist.clamp(self.force_range[0], self.force_range[1])
            if not self.force_is_z:
                self.force_dist[:, -1] = 0

        reward = -self.get_cost(cmd)
        info = self.get_env_infos()
        self.time_step += 1
        self.prev_actions = cmd
        return self.states, reward, info['done'], info

    def render(self,
               env_idx: int=0,
               samples: Optional[Tensor]=None,
               mean: Optional[Tensor]=None,
               state: Optional[Tensor]=None) -> Any:
        return None

    def get_param_dict(self) -> List[Dict[str, Any]]:
        """
        Return a dictionary with parameters to be sent to MPC controller
        """
        param_dict = [dict(t=self.times[self.time_step if self.time_step < len(self.times)-1 else -1],
                           actions=self.actions[idx],
                           ref_dts=self.ref_dts[idx] if not self.ref_dts is None else None,
                           ref_pos=self.ref_pos[idx] if not self.ref_pos is None else None)
                      for idx in range(self.num_envs)]
        return param_dict

    def get_env_description(self) -> List[Tensor]:
        """
        Get a Tensor containing the reference trajectory used to condition policy and value function
        """
        T = 32
        stride = 4
        dim = T//stride

        ref_traj = self.ref_trajectory[:, :, self.time_step:self.time_step+T:stride]
        ref_traj = torch.cat((ref_traj[:, :3], ref_traj[:, 6:10]), dim=1)

        if ref_traj.shape[2] < dim:
            diff = T//stride - ref_traj.shape[2]
            end_step = ref_traj[:, :, -1:].repeat(1, 1, diff)
            ref_traj = torch.cat((ref_traj, end_step), dim=-1)

        cond = ref_traj.reshape(self.num_envs, -1)
        #cond = torch.cat((cond, self.mass[:, None], self.delay_coeff[:, None], self.force_dist), dim=-1)
        info = [cond[i] for i in range(self.num_envs)]
        return info

    def evaluate_success(self, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        num_traj = len(trajectories)
        successes = []
        total_costs = []

        for idx, traj in enumerate(trajectories):
            costs = -traj['rewards']

            total_cost = costs.sum()
            success = True

            successes.append(success)
            total_costs.append(total_cost)
        successes = torch.tensor(successes)
        total_costs = torch.tensor(total_costs)

        success_percentage = torch.sum(successes) / num_traj * 100.
        ret_dict = dict(
            successes=successes,
            total_costs=total_costs,
            success_percentage=success_percentage
        )
        return ret_dict

    def visualize(self, states, ref_traj, dt):
        # Create a new visualizer
        vis = meshcat.Visualizer()
        vis.open()

        vis["/Cameras/default"].set_transform(
            tf.translation_matrix([0, 0, 0]).dot(
                tf.euler_matrix(0, np.radians(-30), -np.pi / 2)))

        vis["/Cameras/default/rotated/<object>"].set_transform(
            tf.translation_matrix([1, 0, 0]))

        vis['/Background'].set_property('top_color', [0, 0, 0])
        vis['/Background'].set_property('bottom_color', [0, 0, 0])

        vis["Quadrotor"].set_object(g.StlMeshGeometry.from_file('./crazyflie2.stl'))

        vertices = np.array([[0, 0.5], [0, 0], [0, 0]]).astype(np.float32)
        vis["lines_segments"].set_object(g.Line(g.PointsGeometry(vertices),
                                                g.MeshBasicMaterial(color=0xff0000, linewidth=100.)))

        vis['ref'].set_object(g.Line(g.PointsGeometry(ref_traj.numpy()[:, :3].T),
                                     g.LineBasicMaterial(color=0xff99ff, linewidth=100.)))

        while True:
            for state in states:
                vis["Quadrotor"].set_transform(
                    tf.translation_matrix([state[0], state[1], state[2]]).dot(
                        tf.quaternion_matrix(state[6:10])))

                vis["lines_segments"].set_transform(
                    tf.translation_matrix([state[0], state[1], state[2]]).dot(
                        tf.quaternion_matrix(state[6:10])))

                time.sleep(dt)