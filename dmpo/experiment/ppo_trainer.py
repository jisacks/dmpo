import torch
import torch.distributions as D
from copy import deepcopy

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.callback import Callback

from torch import Tensor
from torch.nn import Module
from typing import Dict, Any, Optional, List, Union, Tuple, Callable

class PPOTrainer(pl.LightningModule):
    """
    PyTorch Lightning module which implements PPO for updating the policy
    """
    def __init__(self,
                 model: Module,
                 actor_optim: Callable,
                 actor_optim_args: Dict[str, Any],
                 critic_optim: Callable,
                 critic_optim_args: Dict[str, Any],
                 entropy_penalty: float,
                 kl_penalty: float,
                 clip_epsilon: Optional[float] = None,
                 std_clip_epsilon: Optional[float] = None,
                 max_grad_norm: Optional[float] = None,
                 model_subsets: Optional[List[Union[str, List[str]]]] = None,
                 critic_only: bool=False):
        super().__init__()

        # Actor and critic optimizers and associated arguments
        self.optims = [actor_optim, critic_optim]
        self.optims_args = [actor_optim_args, critic_optim_args]
        self.model_subsets = model_subsets

        # PPO parameters
        self.clip_epsilon = clip_epsilon
        self.std_clip_epsilon = std_clip_epsilon
        self.entropy_penalty = entropy_penalty
        self.kl_penalty = kl_penalty
        self.max_grad_norm = max_grad_norm
        self.critic_only = critic_only

        # Turn off PyTorch Lightning automatic optimization
        self.automatic_optimization = False

        # Create a copy of the model for computing CPI loss
        self.model = model
        if hasattr(self.model, 'rollout_task'):
            self.rollout_task = self.model.rollout_task
            self.model.rollout_task = None

            self.old_model = deepcopy(self.model)

            self.model.rollout_task = self.rollout_task
            self.old_model.rollout_task = self.rollout_task
        else:
            self.old_model = deepcopy(self.model)

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        # Get data from batch
        states, actions, rewards, horizons, costs, conds, returns, advantages, means, stds, time_steps = batch
        batch_size, T, H, d_action = horizons.shape

        # Get the optimizers
        actor_opt, critic_opt = self.optimizers()

        # Reset the models
        if hasattr(self.model, 'reset'):
            self.model.reset()
            self.old_model.reset()

        # Iterate over subsequence for actor update
        old_log_probs_list = []; old_std_log_probs_list = []
        log_probs_list = []; std_log_probs_list = []
        entropy_list = []; kl_div_list = []

        for t in range(T):
            state = states[:, t]
            cond = conds[:, t] if conds.ndim > 1 else None
            horizon = horizons[:, t]
            cost = costs[:, t] if costs.ndim > 1 else None
            mean = means[:, t] if means.ndim > 1 else None
            std = stds[:, t] if stds.ndim > 1 else None

            # Get old action log probabilities
            with torch.no_grad():
                self.old_model.to('cuda')
                out = self.old_model(state, cond=cond, costs=cost, mean=mean, std=std)
                old_dist = out['mean_dist']
                old_std_dist = out['std_dist'] if 'std_dist' in out.keys() else None

                old_log_probs = old_dist.log_prob(horizon)
                old_log_probs = old_log_probs.mean(dim=(2)).mean(dim=(1))
                old_log_probs_list.append(old_log_probs)

                if not old_std_dist is None:
                    old_std_log_probs = old_std_dist.log_prob(std)
                    old_std_log_probs = old_std_log_probs.mean(dim=(2)).mean(dim=(1))
                    old_std_log_probs_list.append(old_std_log_probs)

            # Get new action log probabilities
            out = self.model(state, cond=cond, costs=cost, mean=mean, std=std)
            dist = out['mean_dist']
            std_dist = out['std_dist'] if 'std_dist' in out.keys() else None

            log_probs = dist.log_prob(horizon)
            log_probs = log_probs.mean(dim=(2)).mean(dim=(1))
            log_probs_list.append(log_probs)

            if not std_dist is None:
                std_log_probs = std_dist.log_prob(std)
                std_log_probs = std_log_probs.mean(dim=(2)).mean(dim=(1))
                std_log_probs_list.append(std_log_probs)

            # Compute entropy and KL divergence
            if not dist is None:
                entropy = dist.entropy()
                kl_div = D.kl_divergence(dist, old_dist)

                entropy_list.append(entropy)
                kl_div_list.append(kl_div)

        # Stack the lists
        old_log_probs = torch.stack(old_log_probs_list, dim=1)
        log_probs = torch.stack(log_probs_list, dim=1)
        entropies = torch.stack(entropy_list, dim=1)
        kl_divs = torch.stack(kl_div_list, dim=1)

        # Compute the actor loss
        ratio = torch.exp(log_probs - old_log_probs)
        cpi_loss = ratio * advantages.squeeze(2)

        # Compute actor loss for learning STD
        if not std_dist is None:
            old_std_log_probs = torch.stack(old_std_log_probs_list, dim=1)
            std_log_probs = torch.stack(std_log_probs_list, dim=1)
            std_ratio = torch.exp(std_log_probs - old_std_log_probs)
            std_cpi_loss = std_ratio * advantages.squeeze(2)
        else:
            std_ratio = None

        if not self.clip_epsilon is None:
            clip_loss = ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.squeeze(2)
            actor_loss = -torch.min(cpi_loss, clip_loss).mean()

            if not std_ratio is None:
                clip_loss = std_ratio.clamp(1 - self.std_clip_epsilon,
                                            1 + self.std_clip_epsilon) * advantages.squeeze(2)
                actor_loss += -torch.min(std_cpi_loss, clip_loss).mean()
        else:
            actor_loss = -cpi_loss.mean() - std_cpi_loss.mean()

        # Compute entropy penalty and KL divergence
        entropy = entropies.mean()
        kl_div = kl_divs.mean()
        total_loss = actor_loss - self.entropy_penalty * entropy + self.kl_penalty * kl_div

        # Compute ratio deviation
        ratio_dev = torch.abs(ratio - 1).mean()

        # Optimize the actor
        actor_opt.zero_grad()

        if not self.critic_only:
            self.manual_backward(total_loss)
            if not self.max_grad_norm is None:
                torch.nn.utils.clip_grad_norm_(actor_opt.param_groups[0]['params'], self.max_grad_norm)
            actor_opt.step()

        actor_opt.zero_grad()
        critic_opt.zero_grad()

        # Reset the model
        if hasattr(self.model, 'reset'):
            self.model.reset()
            self.old_model.reset()

        # Iterate over subsequence for critic update
        values_list = []
        for t in range(T):
            state = states[:, t]
            cond = conds[:, t] if conds.ndim > 1 else None
            cost = costs[:, t] if costs.ndim > 1 else None
            mean = means[:, t] if means.ndim > 1 else None
            std = stds[:, t] if stds.ndim > 1 else None

            out = self.model(state, cond=cond, costs=cost, mean=mean, std=std)
            value = out['value']
            values_list.append(value)

        # Stack the list
        values = torch.stack(values_list, dim=1)

        # Compute the critic loss
        critic_loss = 0.5 * (returns - values).pow(2).mean()

        # Optimize the critic
        critic_opt.zero_grad()
        self.manual_backward(critic_loss)
        if not self.max_grad_norm is None:
            torch.nn.utils.clip_grad_norm_(critic_opt.param_groups[0]['params'], self.max_grad_norm)
        critic_opt.step()

        actor_opt.zero_grad()
        critic_opt.zero_grad()

        # Reset the model
        if hasattr(self.model, 'reset'):
            self.model.reset()
            self.old_model.reset()

        loss_dict = dict(
            actor_loss=actor_loss.detach(),
            critic_loss=critic_loss.detach(),
            total_loss=total_loss.detach(),
            ratio=ratio.mean().detach(),
            ratio_dev=ratio_dev.detach(),
            entropy=entropy.detach(),
            kl_div=kl_div.detach()
        )

        if not std_ratio is None:
            loss_dict['std_ratio'] = std_ratio.mean().detach()

        batch_dict = {'loss': total_loss, 'log': loss_dict}
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return batch_dict

    def configure_optimizers(self) -> List[Dict[str, Any]]:
        optimizers = []
        for idx, optim in enumerate(self.optims):
            # Get the proper subset of parameters for each optimizer
            if not self.model_subsets is None:
                model_subset = self.model_subsets

                if isinstance(model_subset[idx], str):
                    # If a single attribute is a subset
                    params = filter(lambda x: x.requires_grad, getattr(self.model, model_subset[idx]).parameters())
                else:
                    # If we have a list of attributes for the subset
                    params = []
                    for subset in model_subset[idx]:
                        params.extend(filter(lambda x: x.requires_grad, getattr(self.model, subset).parameters()))
            else:
                # Use all parameters for the optimizer
                params = filter(lambda x: x.requires_grad, self.model.parameters())

            # Create the optimizer
            optimizer = optim(params, **self.optims_args[idx])
            optimizers.append(optimizer)
        opt_dict = [{'optimizer': optimizer} for optimizer in optimizers]
        return opt_dict