import torch
import torch.nn as nn
import torch.distributions as D

from ..models.net_utils import create_net
from ..mpc.task.base_rollout_task import BaseRolloutTask
from .. import utils

from torch import Tensor
from torch.nn import Module
from typing import Dict, Any, Optional, List, Union, Tuple, Callable

STD_MAX = 1e3
STD_MIN = 1e-6

class DMPOPolicy(nn.Module):
    """
    Contains actor and critic for DMPO
    """
    def __init__(self,
                 d_action: int,
                 d_state: int,
                 actor_params: Dict[str, Any],
                 critic_params: Dict[str, Any],
                 shift_params: Optional[Dict[str, Any]] = None,
                 rollout_task: Optional[BaseRolloutTask] = None,
                 horizon: int = 1,
                 num_particles: int = 1,
                 gamma: float = 1.,
                 top_k: int = 1,
                 n_iters: int = 1,
                 init_mean: Union[float, Tensor] = 0.,
                 init_std: Union[float, Tensor] = 1.,
                 sample_params: Optional[Dict[str, Any]] = None,
                 seed_val: int = 0,
                 state_seq_key: str = 'state_seq',
                 mppi_params: Optional[Dict[str, Any]] = None,
                 d_cond: Optional[int] = None,
                 cond_mode: Optional[str] = None,
                 cond_actor: bool = False,
                 cond_critic: bool = False,
                 cond_shift: bool = False,
                 critic_use_cost: bool = False,
                 actor_use_state: bool = False,
                 use_mean: bool = False,
                 mppi_mode: bool = False,
                 is_delta: bool = False,
                 is_gated: bool = False,
                 is_residual: bool = False,
                 mean_search_std: Optional[float] = None,
                 std_search_std: Optional[float] = None,
                 learn_search_std: bool = False,
                 learn_rollout_std: bool = False,
                 action_lows: Optional[Union[float, List[float], Tensor]] = None,
                 action_highs: Optional[Union[float, List[float], Tensor]] = None,
                 state_scale: Optional[Union[float, List[float], Tensor]] = None,
                 cond_scale: Optional[Union[float, List[float], Tensor]] = None,
                 tensor_args: Dict[str, Any] = {'device': 'cpu', 'dtype': torch.float32}):
        """
        :param d_action: Dimensionality of the action space
        :param d_state: Dimensionality of the state space
        :param actor_params: Dictionary of model parameters for the actor network
        :param critic_params: Dictionary of model parameters for the critic network
        :param shift_params: Dictionary of model parameters for the shift model network
        :param rollout_task: Rollout function
        :param horizon: Horizon of MPC controller
        :param num_particles: Number of particles used for rollouts
        :param gamma: Discount factor
        :param top_k: Number of top performing trajectories to save for visualization purposes
        :param n_iters: Number of optimizer iterations
        :param init_mean: Initial mean of the optimizee
        :param init_std: Initial STD of the optimizee
        :param sample_params: Dictionary of parameters for sampling
        :param seed_val: Seed used to generate fixed samples from the standard Gaussian
        :param state_seq_key: Key used to acquire state sequences from dictionary returned by rollout module
        :param mppi_params: Dictionary of MPPI parameters
        :param d_cond: Dimensionality of conditioning variable
        :param cond_mode: Method for incorporating the conditioning variable (e.g. concatenating it to network input)
        :param cond_actor: Flag for conditioning the actor
        :param cond_critic: Flag for conditioning the critic
        :param cond_shift: Flag for conditioning the shift model
        :param critic_use_cost: Flag to indicate that the critic should use the trajectory costs rather than state
        :param actor_use_state: Flag to indicate that the actor should also use state
        :param use_mean: Flag to indicate that we use the mean of our optimizer distribution (for test mode)
        :param mppi_mode: Flag to indicate that we should run vanilla MPPI
        :param is_delta: Flag to indicate that the actor output should be added to current plan (used if not a residual on MPPI)
        :param is_gated: Flag to indicate that we use a gating term on the mean update
        :param is_residual: Flag to indicate that DMPO is learning residuals on MPPI
        :param mean_search_std: Initial STD of the optimizer policy over optimizee means
        :param std_search_std: Initial STD of the optimizer policy over optimizee STDs
        :param learn_search_std: Flag to indicate that we should learn the search STDs
        :param learn_rollout_std: Flag to indicate that we should learn the optimizee STDs (rather than assuming they are fixed)
        :param action_lows: Minimum action values
        :param action_highs: Maximum action values
        :param state_scale: Scale factor on state prior to any network input
        :param cond_scale: Scale factor on conditioning variable prior to any network input
        :param tensor_args: PyTorch Tensor settings
        """
        super().__init__()

        # Rollout parameters
        self.d_action = d_action
        self.d_state = d_state
        self.d_cond = d_cond
        self.horizon = horizon

        self.num_particles = num_particles
        self.gamma = gamma
        self.top_k = top_k
        self.n_iters = n_iters
        self.init_n_iters = n_iters
        self.seed_val = seed_val
        self.state_seq_key = state_seq_key

        # Rollout function
        self.rollout_task = rollout_task

        # Sample parameters
        self.use_halton = sample_params.get('use_halton', True) if not sample_params is None else True

        # MPPI parameters
        self.temperature = mppi_params.get('temperature', 1e-3) if not mppi_params is None else 1e-3
        self.step_size = mppi_params.get('step_size', 1) if not mppi_params is None else 1
        self.scale_costs = mppi_params.get('scale_costs', True) if not mppi_params is None else True

        # Conditioning parameters
        self.cond_mode = cond_mode if not self.d_cond is None else None
        self.cond_actor = cond_actor
        self.cond_critic = cond_critic
        self.cond_shift = cond_shift

        # Other parameters
        self.use_mean = use_mean
        self.mppi_mode = mppi_mode
        self.learn_search_std = learn_search_std
        self.learn_rollout_std = learn_rollout_std
        self.is_gated = is_gated
        self.is_residual = is_residual
        self.is_delta = is_delta
        self.critic_use_cost = critic_use_cost
        self.actor_use_state = actor_use_state

        self.action_lows = utils.to_tensor(action_lows, tensor_args) if not action_lows is None else None
        self.action_highs = utils.to_tensor(action_highs, tensor_args) if not action_highs is None else None
        self.state_scale = utils.to_tensor(state_scale, tensor_args) if not state_scale is None else None
        self.cond_scale = utils.to_tensor(cond_scale, tensor_args) if not cond_scale is None else None
        self.tensor_args = tensor_args

        # Sizes of each input type
        mean_size = self.horizon * self.d_action
        std_size = self.horizon * self.d_action
        cost_size = self.num_particles
        cond_size = self.d_cond

        # Create the shift network
        if not shift_params is None:
            in_size = mean_size
            if learn_rollout_std:
                in_size += std_size
            if cond_mode == 'cat' and not d_cond is None and cond_shift:
                in_size += cond_size

            out_size = self.horizon*self.d_action
            if learn_rollout_std:
                out_size += self.horizon*self.d_action

            self.shift_model = create_net(in_size=in_size, out_size=out_size, **shift_params)
        else:
            self.shift_model = None

        # Create the actor network
        in_size = mean_size + cost_size
        if actor_use_state:
            in_size += d_state
        if learn_rollout_std:
            in_size += std_size
        if cond_mode == 'cat' and not d_cond is None and cond_actor:
            in_size += cond_size

        out_size = mean_size
        if is_gated:
            out_size += mean_size
        if learn_search_std:
            out_size += mean_size
        if learn_rollout_std:
            out_size += std_size
            if learn_search_std:
                out_size += std_size

        self.actor = create_net(in_size=in_size, out_size=out_size, **actor_params)

        # Create the critic
        if not critic_use_cost:
            in_size = d_state + mean_size
        else:
            in_size = mean_size + cost_size

        if learn_rollout_std:
            in_size += std_size
        if cond_mode == 'cat' and not d_cond is None and cond_critic:
            in_size += cond_size

        self.critic = create_net(in_size=in_size, out_size=1, **critic_params)

        # Set the initial mean
        if not isinstance(init_mean, Tensor):
            if isinstance(init_mean, float):
                self.init_mean = init_mean*torch.ones((horizon, d_action), **tensor_args)
            else:
                self.init_mean = torch.tensor(init_mean, **tensor_args).unsqueeze(0).repeat(horizon, 1)
        else:
            self.init_mean = init_mean.to(**tensor_args)

        # Set the initial rollout STD
        if not isinstance(init_std, Tensor):
            if isinstance(init_std, float):
                self.init_std = init_std*torch.ones((horizon, d_action), **tensor_args)
            else:
                self.init_std = torch.tensor(init_std, **tensor_args).unsqueeze(0).repeat(horizon, 1)
        else:
            self.init_std = init_std.to(**tensor_args)

        # Set the initial mean search STD
        if not mean_search_std is None:
            if not isinstance(mean_search_std, Tensor):
                if isinstance(mean_search_std, float):
                    self.mean_search_std = mean_search_std * torch.ones((horizon, d_action), **tensor_args)
                else:
                    self.mean_search_std = torch.tensor(mean_search_std, **tensor_args).unsqueeze(0).repeat(horizon,
                                                                                                            1)
            else:
                self.mean_search_std = mean_search_std.to(**tensor_args)
        else:
            self.mean_search_std = None

        # Set the initial STD search STD
        if not std_search_std is None:
            if not isinstance(std_search_std, Tensor):
                if isinstance(std_search_std, float):
                    self.std_search_std = std_search_std * torch.ones((horizon, d_action), **tensor_args)
                else:
                    self.std_search_std = torch.tensor(std_search_std, **tensor_args).unsqueeze(0).repeat(horizon,
                                                                                                          1)
            else:
                self.std_search_std = std_search_std.to(**tensor_args)
        else:
            self.std_search_std = None

    def set_n_iters(self, n_iters: int):
        self.n_iters = n_iters

    def reset(self):
        self.mean = self.init_mean.clone()
        self.std = self.init_std.clone()
        self.samples = None
        self.state_samples = None
        self.mppi_mean = None
        self.params_stacked = {}

    def update_params(self, kwargs: Dict[str, Any]):
        kwargs_stacked = {key: [] for key in kwargs[0].keys()}
        for kwargs_dict in kwargs:
            for k, v in kwargs_dict.items():
                kwargs_stacked[k].append(v)
        self.params_stacked = kwargs_stacked
        self.rollout_task.update_params(kwargs_stacked)

    def set_task(self, task: BaseRolloutTask) -> None:
        self.rollout_task = task

    def generate_samples(self, batch_size: int):
        num_particles = self.num_particles-1

        if self.samples is None:
            if self.use_halton:
                self.samples = utils.generate_gaussian_halton_samples(num_samples=num_particles,
                                                                      ndims=self.d_action*self.horizon,
                                                                      seed_val=self.seed_val,
                                                                      device=self.tensor_args['device'],
                                                                      dtype=self.tensor_args['dtype'])
            else:
                with torch.random.fork_rng([torch.device(self.tensor_args['device'])]) as rng:
                    torch.random.manual_seed(self.seed_val)
                    self.samples = torch.randn((num_particles, self.d_action*self.horizon),
                                               **self.tensor_args)

            self.samples = self.samples[None, :, :]

        samples = self.samples.view(1, num_particles, self.horizon, self.d_action)
        samples = samples.repeat(batch_size, 1, 1, 1)

        # Always ensure mean is a sample
        zeros = torch.zeros((batch_size, 1, self.horizon, self.d_action), **self.tensor_args)
        samples = torch.cat((zeros, samples), dim=1)

        std = self.std[:, None, :, :]
        samples = samples * std + self.mean[:, None, :, :]

        if not self.action_lows is None and not self.action_highs is None:
            samples = samples.clamp(self.action_lows, self.action_highs)
        return samples

    def run_rollouts(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = x.shape[0]

        with torch.no_grad():
            # Generate action samples
            samples = self.generate_samples(batch_size)

            # Run the rollouts
            trajectories = self.rollout_task.run_rollouts(x, samples)

            # Collect the results
            costs = trajectories['costs']
            states = trajectories[self.state_seq_key]

            if not isinstance(costs, Tensor):
                costs = costs['cost']

            # Compute the total costs
            gamma = torch.tensor([self.gamma ** i for i in range(costs.shape[-1])], **self.tensor_args)
            costs = torch.sum(gamma[None, None, :] * costs, dim=-1)

            # Process states
            self.state_samples = states
            top_values, top_idx = torch.topk(-costs, self.top_k, dim=-1)
            self.top_values = -top_values
            self.top_idx = top_idx
            self.top_trajs = [torch.index_select(states[idx], 0, top_idx[idx]) for idx in range(states.shape[0])]

        return costs, samples

    def process_mean(self, x: Tensor, mean: Tensor):
        batch_size = x.shape[0]
        if mean.ndim == 2:
            mean = mean.unsqueeze(0).repeat(batch_size, 1, 1)
        old_mean = mean

        # Shift mean forward
        shifted_mean = torch.cat((mean[:, 1:], torch.zeros_like(mean[:, -1:])), dim=1)
        return shifted_mean, old_mean

    def process_std(self, x: Tensor, std: Tensor, shift=True):
        batch_size = x.shape[0]
        if std.ndim == 2:
            std = std.unsqueeze(0).repeat(batch_size, 1, 1)
        old_std = std

        # Shift mean forward
        if shift:
            shifted_std = torch.cat((std[:, 1:], std[:, -1:]), dim=1)
            return shifted_std, old_std
        else:
            return old_std

    def get_actor_embedding(self,
                            x: Tensor,
                            costs: Tensor,
                            mean: Optional[Tensor] = None,
                            std: Optional[Tensor] = None,
                            cond: Optional[Tensor] = None) -> Tensor:
        batch_size = x.shape[0]
        if not self.state_scale is None:
           x = x/self.state_scale

        # Compute the normalized costs
        costs = costs.reshape(batch_size, -1)
        cost_mean = costs.mean(dim=-1)
        cost_std = costs.std(dim=-1)
        costs = (costs - cost_mean[:, None]) / (cost_std[:, None] + 1e-6)

        # Reshape the mean and STD
        if not self.action_highs is None and not self.action_lows is None:
           mean = (mean - self.action_lows) / (self.action_highs - self.action_lows)
        mean = mean.view(batch_size, -1)

        if not std is None:
            if not self.action_highs is None and not self.action_lows is None:
                std = std / (self.init_std[0]*10)
            std = std.view(batch_size, -1)

        # Prepare the condition
        if not cond is None and not self.cond_scale is None:
            if self.cond_scale != cond.shape[-1]:
                cond_scale = self.cond_scale[:1].repeat(cond.shape[-1])
                cond = cond / cond_scale
            else:
                cond = cond / self.cond_scale

        # Form the network input
        net_in = torch.cat((costs, mean), dim=-1)
        if self.actor_use_state:
            net_in = torch.cat((net_in, x), dim=-1)
        if self.learn_rollout_std:
            net_in = torch.cat((net_in, std), dim=-1)
        if not cond is None and not self.cond_mode is None and self.cond_actor:
            if self.cond_mode == 'cat':
                net_in = torch.cat((net_in, cond), dim=-1)
            else:
                raise ValueError('Invalid condition mode {} specified.'.format(self.cond_mode))

        return net_in

    def get_critic_embedding(self,
                            x: Tensor,
                            costs: Tensor,
                            mean: Optional[Tensor] = None,
                            std: Optional[Tensor] = None,
                            cond: Optional[Tensor] = None) -> Tensor:

        batch_size = x.shape[0]
        if not self.state_scale is None:
           x = x/self.state_scale

        # Reshape the mean and STD
        if not self.action_highs is None and not self.action_lows is None:
           mean = (mean - self.action_lows) / (self.action_highs - self.action_lows)
        mean = mean.view(batch_size, -1)

        if not std is None:
            if not self.action_highs is None and not self.action_lows is None:
                std = std / (self.init_std[0]*10)
            std = std.view(batch_size, -1)

        # Compute the total costs
        costs = costs.reshape(batch_size, -1)
        cost_mean = costs.mean(dim=-1)
        cost_std = costs.std(dim=-1)
        costs = (costs - cost_mean[:, None]) / (cost_std[:, None] + 1e-6)

        # Prepare the condition
        if not cond is None and not self.cond_scale is None:
            if self.cond_scale != cond.shape[-1]:
                cond_scale = self.cond_scale[:1].repeat(cond.shape[-1])
                cond = cond / cond_scale
            else:
                cond = cond / self.cond_scale

        # Form the network input
        if not self.critic_use_cost:
            net_in = torch.cat((x, mean), dim=-1)
        else:
            net_in = torch.cat((costs, mean), dim=-1)

        if self.learn_rollout_std:
            net_in = torch.cat((net_in, std), dim=-1)

        if not cond is None and not self.cond_mode is None and self.cond_critic:
            if self.cond_mode == 'cat':
                net_in = torch.cat((net_in, cond), dim=-1)
            else:
                raise ValueError('Invalid condition mode {} specified.'.format(self.cond_mode))

        return net_in

    def get_shift_embedding(self,
                            x: Tensor,
                            mean: Optional[Tensor] = None,
                            std: Optional[Tensor] = None,
                            shifted_std: Optional[Tensor] = None,
                            cond: Optional[Tensor] = None) -> Tensor:
        batch_size = x.shape[0]

        # Reshape the mean and STD
        if not self.action_highs is None and not self.action_lows is None:
           mean = (mean - self.action_lows) / (self.action_highs - self.action_lows)
        mean = mean.view(batch_size, -1)

        if not std is None:
            if not shifted_std is None:
                std = std / (shifted_std*10)
            std = std.view(batch_size, -1)

        # Prepare the condition
        if not cond is None and not self.cond_scale is None:
            if self.cond_scale != cond.shape[-1]:
                cond_scale = self.cond_scale[:1].repeat(cond.shape[-1])
                cond = cond / cond_scale
            else:
                cond = cond / self.cond_scale

        # Form the network input
        if self.learn_rollout_std:
            net_in = torch.cat((mean, std), dim=-1)
        else:
            net_in = mean

        if not cond is None and not self.cond_mode is None and self.cond_actor:
            if self.cond_mode == 'cat':
                net_in = torch.cat((net_in, cond), dim=-1)
            else:
                raise ValueError('Invalid condition mode {} specified.'.format(self.cond_mode))

        return net_in

    def run_actor(self,
                  x: Tensor,
                  costs: Tensor,
                  mean: Optional[Tensor] = None,
                  std: Optional[Tensor] = None,
                  cond: Optional[Tensor] = None) -> Tuple[D.Distribution, D.Distribution]:

        # Compute the MPPI update
        if self.mppi_mode or self.is_residual:
            with torch.no_grad():
                mppi_costs = costs
                if self.scale_costs:
                    min_costs = mppi_costs.min(dim=-1)[0][:, None]
                    max_costs = mppi_costs.max(dim=-1)[0][:, None]
                    mppi_costs = (mppi_costs - min_costs)/(max_costs - min_costs + 1e-6)

                weights = torch.softmax(-mppi_costs / self.temperature, dim=1)
                samples = self.generate_samples(weights.shape[0])
                update = torch.sum(samples * weights[:, :, None, None], dim=1)

            mppi_mean = (1 - self.step_size) * self.mean + self.step_size * update.view(-1, self.horizon, self.d_action)
        else:
            mppi_mean = None
        self.mppi_mean = mppi_mean

        # Compute the DMPO update
        if not self.mppi_mode:

            # Get the actor embedding
            net_in = self.get_actor_embedding(x, costs=costs, mean=mean, std=std, cond=cond)

            # Run the actor
            out = self.actor(net_in)
            mean = out[:, :self.horizon*self.d_action].view(-1, self.horizon, self.d_action)
            idx = 1

            # Scale the actor output so that it is in a reasonable range
            if not self.action_lows is None and not self.action_highs is None:
                mean = torch.tanh(mean)
                mean = mean*(self.action_highs - self.action_lows)

            # Get the gating term if used
            if self.is_gated:
                gating = out[:, self.horizon*self.d_action*idx:self.horizon*self.d_action*(idx+1)]
                gating = torch.tanh(gating).view(-1, self.horizon, self.d_action)
                idx += 1

            # Get the updated search STD for the optimizee mean
            if self.learn_search_std:
                log_mean_std = out[:, self.horizon*self.d_action*idx:self.horizon*self.d_action*(idx+1)]
                mean_std = log_mean_std.exp().view(-1, self.horizon, self.d_action)
                mean_std = torch.clamp(self.mean_search_std[None, :, :]*mean_std, STD_MIN, STD_MAX)
                idx += 1
            else:
                mean_std = self.mean_search_std

            # Handle updates to the optimizee STD if used
            if self.learn_rollout_std:

                # Compute the updated optimizee STD
                log_std = out[:, self.horizon*self.d_action*idx:self.horizon*self.d_action*(idx+1)]
                std = log_std.exp().view(-1, self.horizon, self.d_action)
                std = torch.clamp(self.init_std[None, :, :]*std, STD_MIN, STD_MAX)
                idx += 1

                # Get the updated search STD for the optimizee STD
                if self.learn_search_std:
                    log_std_std = out[:, self.horizon*self.d_action*idx:self.horizon*self.d_action*(idx+1)]
                    std_std = log_std_std.exp().view(-1, self.horizon, self.d_action)
                    std_std = torch.clamp(self.std_search_std*std_std, STD_MIN, STD_MAX)
                else:
                    std_std = self.std_search_std

            # Compute the updated mean
            if not self.is_residual and self.is_delta:
                # If not in residual mode and learning a delta on current plan, optionally with a gating term
                if self.is_gated:
                    mean = (1-gating)*self.mean + gating*mean
                else:
                    mean = mean + self.mean
            elif self.is_residual:
                # If in residual mode, form the update on the MPPI proposed mean, optionally using a gating term
                if self.is_gated:
                    mean = (1-gating)*mppi_mean + gating*mean
                else:
                    mean = mppi_mean + mean

            mean_dist = D.Normal(mean, mean_std)
            if self.learn_rollout_std:
                std_dist = D.Normal(std, std_std)
            else:
                std_dist = None
        else:
            mean_dist = D.Normal(mppi_mean, self.std)
            std_dist = None

        return mean_dist, std_dist

    def run_critic(self,
                   x: Tensor,
                   costs: Tensor,
                   mean: Optional[Tensor] = None,
                   std: Optional[Tensor] = None,
                   cond: Optional[Tensor] = None) -> Tensor:

        # Get the critic embedding
        net_in = self.get_critic_embedding(x, costs=costs, mean=mean, std=std, cond=cond)

        # Run the critics
        return self.critic(net_in)

    def run_shift_model(self,
                        x: Tensor,
                        mean: Optional[Tensor] = None,
                        shifted_mean: Optional[Tensor] = None,
                        std: Optional[Tensor] = None,
                        shifted_std: Optional[Tensor] = None,
                        cond: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # Get the shift model embedding
        net_in = self.get_shift_embedding(x, mean=mean, std=std, shifted_std=shifted_std, cond=cond)

        # Run the shift model
        out = self.shift_model(net_in)

        if self.learn_rollout_std:
            mean, log_std = torch.split(out, self.horizon*self.d_action, dim=-1)
            mean = mean.view(-1, self.horizon, self.d_action)
            log_std = log_std.view(-1, self.horizon, self.d_action)
        else:
            mean = out.view(-1, self.horizon, self.d_action)

        # Ensure network output is in a reasonable range
        if not self.action_lows is None and not self.action_highs is None:
            mean = torch.tanh(mean)
            mean = mean * (self.action_highs - self.action_lows)

        # Update the shifted optimizee STD
        if self.learn_rollout_std:
            std = log_std.exp().view(-1, self.horizon, self.d_action)
            std = torch.clamp(shifted_std*std, STD_MIN, STD_MAX)

        # Optionally set the optimizee mean as a residual on the shift-forward mean
        if self.is_residual:
            mean = mean + shifted_mean

        # Return the shifted mean and STD
        if self.learn_rollout_std:
            return mean, std
        else:
            return mean

    def forward(self,
                x: Tensor,
                costs: Optional[Tensor] = None,
                mean: Optional[Tensor] = None,
                std: Optional[Tensor] = None,
                cond: Optional[Tensor] = None,
                run_critic: bool = True,
                **kwargs) -> Dict[str, Any]:
        x = utils.to_tensor(x, self.tensor_args)

        # Handle the mean input
        if not mean is None:
            self.mean = mean
        shifted_mean, old_mean = self.process_mean(x, self.mean)
        self.mean = shifted_mean

        # Handle the STD input
        if self.learn_rollout_std and not self.mppi_mode:
            if not std is None:
                self.std = std
            shifted_std, old_std = self.process_std(x, self.std)
            self.std = shifted_std
        else:
            old_std = self.process_std(x, self.std, shift=False)
            self.std = old_std

        # Use learned shift model
        if not self.shift_model is None and not self.mppi_mode:
            if self.learn_rollout_std:
                self.mean, self.std = self.run_shift_model(x, old_mean, self.mean, old_std, self.std, cond)
            else:
                self.mean = self.run_shift_model(x, old_mean,self.mean, old_std, self.std, cond)

        # Run the specified # of iterations of optimization
        for iter in range(self.n_iters):

            # Ensure we reoptimize if running multiple iterations
            if iter > 0:
                costs = None
                self.rollout_task.update_params(self.params_stacked)

            # Run rollouts to compute costs
            if costs is None:
                costs, samples = self.run_rollouts(x)
            else:
                samples = None

            # Run the actor
            mean_dist, std_dist = self.run_actor(x,
                                                 costs=costs,
                                                 mean=self.mean,
                                                 std=self.std,
                                                 cond=cond)

            # Sample the mean
            if self.use_mean:
                horizon = mean_dist.loc
            else:
                horizon = mean_dist.rsample()
            horizon = horizon.clamp(self.action_lows, self.action_highs)
            self.mean = horizon

            # Sample the STD
            if self.learn_rollout_std and not self.mppi_mode:
                if self.use_mean:
                    self.std = std_dist.loc
                else:
                    self.std = std_dist.rsample()
                self.std = self.std.clamp(STD_MIN, STD_MAX)

        # Set the action to be the first in the horizon
        action = horizon[:, 0]

        # Run the critic
        if self.mppi_mode or not run_critic:
            value = torch.zeros(x.shape[0])
        else:
            value = self.run_critic(x,
                                    costs=costs,
                                    mean=old_mean,
                                    std=old_std,
                                    cond=cond)

        return dict(
            action=action,
            horizon=horizon,
            costs=costs,
            mean=self.mean,
            old_mean=old_mean,
            std=self.std,
            old_std=old_std,
            samples=samples,
            mean_dist=mean_dist,
            std_dist=std_dist,
            value=value
        )