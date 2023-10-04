import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, Sampler, Sequence
import numpy as np

from .. import utils

from torch import Tensor
from torch.nn import Module
from typing import Dict, Any, Optional, List, Union, Tuple, Callable, Iterator

class SubsetSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for i in self.indices:
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)

class DatasetBuffer(Dataset):
    """
    Dataset buffer class for PPO
    """
    def __init__(self, discount: float=1, seq_length: int=1, stride: int=1, gae_lambda: Optional[float]=None) -> None:
        """
        :param discount: Discount factor
        """
        super().__init__()
        self.discount = discount
        self.seq_length = seq_length
        self.stride = stride
        self.gae_lambda = gae_lambda

        self.clear()

    def clear(self) -> None:
        self.states = []
        self.actions = []
        self.horizons = []
        self.costs = []
        self.conds = []
        self.rewards = []
        self.values = []
        self.stds = []
        self.returns = []
        self.advantages = []
        self.means = []
        self.time_steps = []

    def push(self, trajectories: List[Dict[str, Any]]) -> None:
        # Iterate through the trajectories
        for traj in trajectories:
            self.states.append(traj['states'])
            self.actions.append(traj['actions'])
            self.costs.append(traj['costs'])
            self.rewards.append(traj['rewards'])
            self.values.append(traj['values'])
            self.conds.append(traj['conds'])
            self.horizons.append(traj['horizons'])
            self.means.append(traj['means'])
            self.stds.append(traj['stds'])
            self.time_steps.append(traj['time_steps'])

    def __len__(self) -> int:
        return len(self.states)

    def compute_returns_and_advantages(self) -> None:
        if self.gae_lambda is None:
            for traj_idx in range(len(self)):
                rewards = self.rewards[traj_idx]
                values = self.values[traj_idx]

                R = values[-1]
                returns = []
                for reward in reversed(rewards):
                    R = self.discount*R + reward
                    returns.insert(0, R)
                returns = torch.stack(returns)
                self.returns.append(returns)

            returns = torch.stack(self.returns)
            values = torch.stack(self.values)
            adv = returns - values
        else:
            for traj_idx in range(len(self)):
                rewards = self.rewards[traj_idx]
                values = self.values[traj_idx]

                last_gae_lam = 0
                advantages = []
                for t in reversed(range(len(rewards))):
                    if t == len(rewards)-1:
                        next_values = values[-1]
                    else:
                        next_values = values[t+1]
                    delta = rewards[t] + self.discount*next_values - values[t]
                    last_gae_lam = delta + self.discount*self.gae_lambda*last_gae_lam
                    advantages.insert(0, last_gae_lam)
                advantages = torch.stack(advantages)
                returns = advantages + values
                self.advantages.append(advantages)
                self.returns.append(returns)
            adv = torch.stack(self.advantages)

        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        self.advantages = adv

    def split_into_subsequences(self) -> None:
        states = self.states
        actions = self.actions
        horizons = self.horizons
        costs = self.costs
        conds = self.conds
        returns = self.returns
        rewards = self.rewards
        advantages = self.advantages
        means = self.means
        stds = self.stds
        time_steps = self.time_steps

        self.clear()

        for traj_idx in range(len(states)):
            ep_length = states[traj_idx].shape[0]
            indices = np.arange(0, ep_length-self.seq_length+1, self.stride)
            n_subseqs = len(indices)

            # Split into subsequences
            self.states.extend(
                [states[traj_idx][i:i+self.seq_length] for i in indices])
            self.actions.extend(
                [actions[traj_idx][i:i+self.seq_length] for i in indices])
            self.rewards.extend(
                [rewards[traj_idx][i:i+self.seq_length] for i in indices])
            self.time_steps.extend(
                [time_steps[traj_idx][i:i+self.seq_length] for i in indices])

            # Handle returns and advantages if computed
            if len(returns) > 0:
                self.returns.extend(
                    [returns[traj_idx][i:i+self.seq_length] for i in indices])
                self.advantages.extend(
                    [advantages[traj_idx][i:i+self.seq_length] for i in indices])
            else:
                self.returns.extend([None for _ in range(n_subseqs)])
                self.advantages.extend([None for _ in range(n_subseqs)])

            # Append the full sampled plans
            if not horizons[traj_idx] is None:
                self.horizons.extend(
                    [horizons[traj_idx][i:i+self.seq_length] for i in indices])
            else:
                self.horizons.extend([None for _ in range(n_subseqs)])

            # Handle the optional means variables
            if not means[traj_idx] is None:
                self.means.extend(
                    [means[traj_idx][i:i+self.seq_length] for i in indices])
            else:
                self.means.extend([None for _ in range(n_subseqs)])

            # Handle the optional stds variables
            if not stds[traj_idx] is None:
                self.stds.extend(
                    [stds[traj_idx][i:i + self.seq_length] for i in indices])
            else:
                self.stds.extend([None for _ in range(n_subseqs)])

            # Handle the optional costs variable
            if not costs[traj_idx] is None:
                self.costs.extend(
                    [costs[traj_idx][i:i+self.seq_length] for i in indices])
            else:
                self.costs.extend([None for _ in range(n_subseqs)])

            # Handle the optional condition variable
            if not conds[traj_idx] is None:
                self.conds.extend(
                    [conds[traj_idx][i:i+self.seq_length] for i in indices])
            else:
                self.conds.extend([None for _ in range(n_subseqs)])


    def get_samplers(self):
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        sampler = SubsetSampler(indices)
        return sampler

    def __getitem__(self, idx):
        return self.states[idx], \
               self.actions[idx], \
               self.rewards[idx], \
               self.horizons[idx] if not self.horizons[idx] is None else torch.empty(()), \
               self.costs[idx] if not self.costs[idx] is None else torch.empty(()), \
               self.conds[idx] if not self.conds[idx] is None else torch.empty(()), \
               self.returns[idx] if not self.returns[idx] is None else torch.empty(()), \
               self.advantages[idx] if not self.advantages[idx] is None else torch.empty(()), \
               self.means[idx] if not self.means[idx] is None else torch.empty(()), \
               self.stds[idx] if not self.stds[idx] is None else torch.empty(()), \
               self.time_steps[idx]
