import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np

from .experiment import Experiment
from .ppo_rollout import ppo_rollout
from ..dataset.dataset_buffer import DatasetBuffer
from .experiment_utils import create_task, create_env, create_optim
from .ppo_trainer import PPOTrainer
from .. import utils

from torch import Tensor
from torch.nn import Module
from typing import Dict, Any, Optional, List, Union, Tuple, Callable

class PPOExperiment(Experiment):
    """
    Experiment class for running PPO
    """
    def __init__(self,
                 env_name: str,
                 env_config: Dict[str, Any],
                 model_config: Dict[str, Any],
                 actor_optim_config: Dict[str, Any],
                 critic_optim_config: Dict[str, Any],
                 task_config: Optional[Dict[str, Any]]=None,
                 batch_size: int=1,
                 seed: int=0,
                 n_iters: int=1,
                 use_condition: bool=False,
                 train_episode_len: int=1,
                 val_episode_len: int=1,
                 train_episodes: int=1,
                 val_episodes: int=1,
                 n_val_envs: Optional[int]=None,
                 break_if_done: bool=False,
                 dynamic_env: bool=False,
                 num_workers: int=0,
                 model_file: Optional[str]=None,
                 n_epochs: int=1,
                 n_gpus: int=1,
                 log_folder: str='logs',
                 exp_name: str='experiment',
                 dtype: str='float',
                 device: str='cuda',
                 env_device: str='cpu',
                 val_every: int=1,
                 max_grad_norm: float=1,
                 n_pretrain_steps: Optional[int]=None,
                 n_pretrain_epochs: int=1,
                 dataset_config: Dict[str, Any]={},
                 trainer_config: Dict[str, Any]={},
                 train_env_config: Dict[str, Any]={},
                 val_env_config: Dict[str, Any]={}) -> None:
        """
        :param env_name: Environment to load
        :param env_config: Configuration for environment
        :param model_config: Configuration for actor\critic model
        :param actor_optim_config: Configuration for actor optimizer
        :param critic_optim_config: Configuration for critic optimizer
        :param task_config: Configuration for the MPC task
        :param batch_size: Batch size
        :param seed: Base seed for rollouts
        :param n_iters: # of iterations of rollout-train-test
        :param use_condition: Handle logic for forming and sending conditioning variable to policy and critic
        :param train_episode_len: Max length of training episodes
        :param val_episode_len: Max length of validation episodes
        :param train_episodes: # of train episodes per iteration
        :param val_episodes: # of validation episodes per iteration
        :param n_val_envs: # of validation environments (can be different than # of training envs)
        :param break_if_done: Break from rollout if all environments have finished
        :param dynamic_env: Environment is dynamic (requires additional processing for MPC)
        :param num_workers: # of threads to use for loading training data
        :param model_file: Optional pre-trained model to load
        :param n_epochs: # of epochs to train for each iteration
        :param n_gpus: # of GPU devices to use
        :param log_folder: Folder to use for logging and checkpoints
        :param exp_name: Name of experiment for logging
        :param dtype: PyTorch data type
        :param device: PyTorch device
        :param val_every: # of epochs per which we run on validation data
        :param max_grad_norm: Maximum norm used for gradient clipping
        :param n_pretrain_steps: # of iterations to pretrain critic
        :param n_pretrain_epochs: # of epochs per iteration for pretraining critic
        :param dataset_config: Configuration for dataset buffer
        :param trainer_config: Configuration for the trainer
        :param train_env_config: Additional environment configuration for training rollouts
        :param val_env_config: Additional environment configuration for validation rollouts
        """
        super().__init__(model_config=model_config,
                         optim_config=actor_optim_config,
                         n_epochs=n_epochs,
                         log_folder=log_folder,
                         exp_name=exp_name,
                         dtype=dtype,
                         device=device)

        self.batch_size = batch_size
        self.seed = seed
        self.n_iters = n_iters
        self.use_condition = use_condition
        self.train_episode_len = train_episode_len
        self.val_episode_len = val_episode_len
        self.train_episodes = train_episodes
        self.val_episodes = val_episodes
        self.val_every = val_every
        self.n_val_envs = n_val_envs
        self.break_if_done = break_if_done
        self.dynamic_env = dynamic_env
        self.num_workers = num_workers
        self.n_pretrain_steps = n_pretrain_steps
        self.n_pretrain_epochs = n_pretrain_epochs
        self.model_config = model_config
        self.task_config = task_config
        self.train_env_config = train_env_config
        self.val_env_config = val_env_config

        self.best_percent = None
        self.best_cost = None

        # Optionally load the model
        if not model_file is None:
            model_dict = torch.load(model_file, map_location='cpu')
            state_dict = model_dict['state_dict']
            self.model.load_state_dict(state_dict)
            self.model.to(**self.tensor_args)

        if hasattr(self.model, 'horizon'):
            self.horizon = self.model.horizon
        self.d_action = self.model.d_action

        # Get the critic optimizer
        self.critic_optim, self.critic_optim_args = create_optim(critic_optim_config)

        # Create the environment
        self.env_tensor_args = {'device':env_device, 'dtype':self.dtype}
        self.env = create_env(env_name, tensor_args=self.env_tensor_args, **env_config)

        self.model.action_lows = utils.to_tensor(self.env.action_lows, self.tensor_args)
        self.model.action_highs = utils.to_tensor(self.env.action_highs, self.tensor_args)

        # Create the task
        if not task_config is None:
            self.task = create_task(env_name=env_name,
                                    num_envs=self.env.num_envs,
                                    tensor_args=self.tensor_args,
                                    **task_config)
            self.model.set_task(self.task)
        else:
            self.task = None

        # Create the dataset buffer
        self.dataset = DatasetBuffer(**dataset_config)

        # Create the PPO model trainer
        self.model_trainer = PPOTrainer(model=self.model,
                                        actor_optim=self.optim,
                                        actor_optim_args=self.optim_args,
                                        critic_optim=self.critic_optim,
                                        critic_optim_args=self.critic_optim_args,
                                        max_grad_norm=max_grad_norm,
                                        **trainer_config)

        # Create the PyTorch Lightning Trainer instance
        self.trainer = pl.Trainer(devices=n_gpus,
                                  accelerator='gpu',
                                  logger=self.logger,
                                  max_epochs=self.n_epochs,
                                  check_val_every_n_epoch=val_every,
                                  num_sanity_val_steps=0)

    def run_rollouts(self, itr: int=0, is_train: bool=False):
        if is_train:
            # Set the training configuration
            n_episodes = self.train_episodes
            ep_length = self.train_episode_len
            self.env.set_param(self.train_env_config)

            # Use a different random seed for each training trial
            seed = self.seed + 123*itr
        else:
            # Set the validation configuration
            n_episodes = self.val_episodes
            ep_length = self.val_episode_len
            self.env.set_param(self.val_env_config)

            # Use a fixed random seed for each validation trial
            seed = self.seed

            # Change # of environments if different for validation
            if not self.n_val_envs is None:
                n_envs = self.env.num_envs
                self.env.num_envs = self.n_val_envs

        # Validation mode uses the mean optimizer update
        if not is_train:
            use_mean = self.model.use_mean
            self.model.use_mean = True

        # Perform the rollout
        with torch.no_grad():
            trajectories = ppo_rollout(env=self.env,
                                       model=self.model,
                                       n_episodes=n_episodes,
                                       ep_length=ep_length,
                                       base_seed=seed,
                                       break_if_done=self.break_if_done,
                                       use_condition=self.use_condition,
                                       dynamic_env=self.dynamic_env,
                                       use_tqdm=True,
                                       tensor_args=self.tensor_args)

        # Reset parameters if modified during validation mode
        if not is_train:
            self.model.use_mean = use_mean

            if not self.n_val_envs is None:
                self.env.num_envs = n_envs

        # Compute statistics
        success_dict = self.env.evaluate_success(trajectories)
        stat_dict = utils.compute_statistics(success_dict)

        # Display statistics
        success_percentage = success_dict['success_percentage']
        mean_cost = stat_dict['mean_cost']
        std_cost = stat_dict['std_cost']
        mean_success_cost = stat_dict['mean_success_cost']
        std_success_cost = stat_dict['std_success_cost']

        # Save current model
        utils.make_dir(self.logger.log_dir + '/checkpoints')

        filename = self.logger.log_dir + '/checkpoints/last.pt'
        torch.save(dict(state_dict=self.model.state_dict(),
                        model_config=self.model_config,
                        task_config=self.task_config), filename)

        # Save model if validation performance better in terms of cost
        if not is_train:
            if (self.best_cost is None) or (self.best_cost >= mean_cost):
                self.best_percent = success_percentage
                self.best_cost = mean_cost
                self.model.reset()

                utils.make_dir(self.logger.log_dir + '/checkpoints')
                filename = self.logger.log_dir + '/checkpoints/best.pt'
                torch.save(dict(state_dict=self.model.state_dict(),
                                model_config=self.model_config,
                                task_config=self.task_config),
                           filename)

        # Log validation performance
        if not is_train:
            self.logger.experiment.add_scalar('Best Success Percentage', self.best_percent, itr)
            self.logger.experiment.add_scalar('Best Mean Cost', self.best_cost, itr)

            self.logger.experiment.add_scalar('Success Percentage', success_percentage, itr)
            self.logger.experiment.add_scalar('Mean Cost', mean_cost, itr)
            self.logger.experiment.add_scalar('Mean Success Cost', mean_success_cost, itr)

        print('Success Metric (Best) = {:.2f} ({:.2f}), '
              'Mean Cost (Best) = {:.3e} +/- {:.3e} ({:.3e}), '
              'Mean Success Cost = {:.3e} +/- {:.3e}'.format(success_percentage,
                                                             self.best_percent if not self.best_percent is None else 0,
                                                             mean_cost,
                                                             std_cost,
                                                             self.best_cost if not self.best_percent is None else np.inf,
                                                             mean_success_cost,
                                                             std_success_cost))

        return trajectories

    def run(self) -> None:
        # Main loop
        for itr in range(self.n_iters):
            print('Iteration {}'.format(itr))

            # Collect training data
            trajectories = self.run_rollouts(itr=itr, is_train=True)

            # Add trajectories to dataset
            self.dataset.clear()
            self.dataset.push(trajectories)
            self.dataset.compute_returns_and_advantages()
            self.dataset.split_into_subsequences()

            # Fit the model
            sampler = self.dataset.get_samplers()
            data_loader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     sampler=sampler,
                                     num_workers=self.num_workers)

            # Reset the PyTorch Lightning trainer
            self.trainer.fit_loop.epoch_progress.reset()

            # Remove the current checkpoints
            utils.remove_file(self.trainer.checkpoint_callback.best_model_path)

            last_model_path = '/'.join(self.trainer.checkpoint_callback.best_model_path.split('/')[:-1])
            last_model_path = last_model_path + '/last.ckpt'
            utils.remove_file(last_model_path)

            # Reset the PyTorch Lightning checkpoint callback
            self.trainer.checkpoint_callback.best_k_models = {}
            self.trainer.checkpoint_callback.best_model_score = None
            self.trainer.checkpoint_callback.best_model_path = ''
            self.trainer.checkpoint_callback.filename = None

            # Set up for pretraining
            if not self.n_pretrain_steps is None and itr < self.n_pretrain_steps:
                self.model_trainer.critic_only = True
                self.trainer.fit_loop.max_epochs = self.n_pretrain_epochs
            else:
                self.model_trainer.critic_only = False
                self.trainer.fit_loop.max_epochs = self.n_epochs

            # Train the model
            self.model_trainer.old_model.load_state_dict(self.model.state_dict())
            self.trainer.fit(self.model_trainer, data_loader)

            # Test current policy
            self.model.to(**self.tensor_args)
            self.model.eval()

            if itr%self.val_every == 0:
                self.run_rollouts(itr=itr, is_train=False)
