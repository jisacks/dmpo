import torch
import numpy as np
from tqdm import tqdm
from time import time

from .. import utils

from torch.nn import Module
from typing import Dict, Any, Optional, List

def ppo_rollout(env,
                model: Module,
                n_episodes: int=1,
                ep_length: int=1,
                base_seed: int=0,
                break_if_done: bool=True,
                use_condition: bool=False,
                dynamic_env: bool=False,
                use_tqdm: bool=True,
                save_data: bool=True,
                run_critic=True,
                tensor_args: Dict[str, Any]={'device': 'cpu', 'dtype': torch.float32}) -> List[Dict[str, Any]]:
    """
    Rollout controller in environment
    :param env: Environment to rollout in
    :param model: Container of actor to rollout and critic
    :param n_episodes: # of episodes
    :param ep_length: Episode length
    :param base_seed: Base seed (which is modified per episode)
    :param break_if_done: Break from episode loop if done signal has been achieved
    :param use_condition: Extract conditional information from environment to give to the policy
    :param dynamic_env: Run a dynamic environment
    :param save_data: Flag to indicate if we should return all rollout data
    :param run_critic: Flag to indicate if we should run the critic
    :param tensor_args: PyTorch Tensor settings
    """
    trajectories = []

    for ep in range(n_episodes):

        # Set the seed
        episode_seed = base_seed + 12345*ep
        np.random.seed(episode_seed)
        torch.random.manual_seed(episode_seed)
        if tensor_args['device'] == 'cuda':
            torch.cuda.manual_seed(episode_seed)

        # Reset the environment
        env.reset()
        param_dict = env.get_param_dict()

        # Optional settings if using MPC under the hood
        if hasattr(model, 'update_params'):
            model.update_params(param_dict)

        if hasattr(model, 'set_seed'):
            model.set_seed(episode_seed)

        # Retrieve environment information for conditioning model
        if use_condition:
            env_info = env.get_env_description()
            cond = torch.stack(env_info, dim=0).to(**tensor_args)

            if hasattr(model, 'set_cond'):
                model.set_cond(env_info)
        else:
            cond = None

        # Reset the actor and critic
        if hasattr(model, 'reset'):
            model.reset()

        # Perform an episode
        state_list = []; cond_list = []; reward_list = []; info_list = []
        action_list = []; horizon_list = []; costs_list = []; value_list = []
        old_mean_list = []; old_std_list = []
        time_step_list = []

        pbar = tqdm(range(ep_length)) if use_tqdm else range(ep_length)
        for t in pbar:
            # Get current state
            curr_states = env.get_env_state()

            if hasattr(env, 'get_env_obs'):
                curr_obs = env.get_env_obs()
            else:
                curr_obs = curr_states

            # Run the actor and critic
            out = model(curr_obs, cond=cond, run_critic=run_critic)

            # Retrieve model outputs
            action = out['action']
            value = out['value']
            horizon = out['horizon'] if 'horizon' in out.keys() else None
            old_mean = out['old_mean'] if 'old_mean' in out.keys() else None
            old_std = out['old_std'] if 'old_std' in out.keys() else None
            costs = out['costs'] if 'costs' in out.keys() else None

            # Prepare action to be applied
            if isinstance(action, list) and isinstance(action[0], torch.Tensor):
                action = torch.stack(action).cpu()

            # Take a step in the environment
            with torch.no_grad():
                obs, reward, done, info = env.step(action)

            # Store information
            state_list.append(torch.stack(curr_states).cpu() if not isinstance(curr_states, torch.Tensor) else curr_states.cpu())
            action_list.append(action.cpu())
            reward_list.append(reward.cpu())
            info_list.append(info)

            if save_data:
                value_list.append(value.cpu())
                time_step_list.append(t)

                if not horizon is None:
                    horizon_list.append(horizon.cpu())

                if not cond is None:
                    cond_list.append(cond.cpu())

                if not costs is None:
                    costs_list.append(costs.cpu())
                else:
                    costs_list.append(None)

                if not old_mean is None:
                    old_mean_list.append(old_mean.cpu())
                else:
                    old_mean_list.append(None)

                if not old_std is None:
                    old_std_list.append(old_std.cpu())
                else:
                    old_std_list.append(None)

            # Handle a dynamic environment
            if dynamic_env:
                param_dict = env.get_param_dict()
                if hasattr(model, 'update_params'):
                    model.update_params(param_dict)

                if use_condition:
                    env_info = env.get_env_description()
                    cond = torch.stack(env_info, dim=0).to(**tensor_args)

                    if hasattr(model, 'set_cond'):
                        model.set_cond(env_info)

            # Break if done
            if break_if_done and done:
                break

        # Store trajectories
        states = utils.stack_list_tensors(state_list)
        actions = utils.stack_list_tensors(action_list)
        rewards = utils.stack_list_tensors(reward_list)
        infos = utils.transpose_dict_list(info_list)

        if save_data:
            horizons = utils.stack_list_tensors(horizon_list)
            values = utils.stack_list_tensors(value_list)
            time_steps = [torch.tensor(time_step_list).unsqueeze(1) for _ in range(env.num_envs)]

            if not cond is None:
                conds = utils.stack_list_tensors(cond_list)
            else:
                conds = None

            if not costs_list[0] is None:
                costs = utils.stack_list_tensors(costs_list)
            else:
                costs = None

            if not old_mean_list[0] is None:
                old_means = utils.stack_list_tensors(old_mean_list)
            else:
                old_means = None

            if not old_std_list[0] is None:
                old_stds = utils.stack_list_tensors(old_std_list)
            else:
                old_stds = None

        for idx in range(env.num_envs):
            if save_data:
                traj = dict(
                    states=states[idx],
                    rewards=rewards[idx],
                    infos=infos[idx],
                    actions=actions[idx],
                    horizons=horizons[idx],
                    costs=costs[idx] if not costs is None else None,
                    values=values[idx],
                    conds=conds[idx] if not conds is None else None,
                    means=old_means[idx] if old_means is not None else None,
                    stds=old_stds[idx] if not old_stds is None else None,
                    time_steps=time_steps[idx]
                )
            else:
                traj = dict(
                    states=states[idx],
                    actions=actions[idx],
                    rewards=rewards[idx],
                    infos=infos[idx]
                )
            trajectories.append(traj)
    return trajectories
