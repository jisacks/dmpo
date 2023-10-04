import torch

from dmpo.envs.quadrotor import QuadrotorEnv
from dmpo.experiment.ppo_rollout import ppo_rollout
from dmpo.experiment.experiment_utils import create_model, create_task
from dmpo import utils

if __name__ == '__main__':
    # Main script configuration
    mpc_file = '../config/mpc/quadrotor_zigzagyaw_mppi.yml'
    train_file = '../config/experiments/quadrotor_dmpo_zigzagyaw.yml'
    model_file = 'quadrotor_logs/quadrotor_dmpo_zigzagyaw/version_0/checkpoints/best.pt'
    use_gpu = True
    is_mppi = True

    # Set up PyTorch
    device = 'cuda' if use_gpu else 'cpu'
    env_tensor_args = {'device': device, 'dtype': torch.double}
    ctrl_tensor_args = {'device': device, 'dtype': torch.double}

    # Get environment
    exp_params = utils.load_yaml(mpc_file)
    rollout_params = exp_params['rollout']
    env_params = exp_params['environment']
    env_cost_params = exp_params['env_cost']
    cost_params = exp_params['cost']

    env_type = rollout_params.pop('env_type')
    num_envs = rollout_params.pop('num_envs')
    env = QuadrotorEnv(num_envs=num_envs, tensor_args=env_tensor_args, **env_params)

    # Instantiate controller from configuration
    if not is_mppi:
        model_dict = torch.load(model_file, map_location='cpu')

        state_dict = model_dict['state_dict']
        model_config = model_dict['model_config']

        train_config = utils.load_yaml(train_file)
        task_config = train_config['task_config']
    else:
        train_config = utils.load_yaml(train_file)
        model_config = train_config['model_config']
        task_config = train_config['task_config']

    if is_mppi:
        model_config['mppi_mode'] = True
        model_config['horizon'] = 32
        model_config['num_particles'] = 2048
        model_config['n_iters'] = 1
        task_config['task_config_file'] = mpc_file

    model = create_model(tensor_args=ctrl_tensor_args, **model_config)
    if not is_mppi:
        model.load_state_dict(state_dict)
    model.to(**ctrl_tensor_args)

    model.action_lows = utils.to_tensor(env.action_lows, ctrl_tensor_args)
    model.action_highs = utils.to_tensor(env.action_highs, ctrl_tensor_args)
    model.use_mean = True

    # Create the MPC task for performing rollouts
    task = create_task(env_name=env_type,
                       num_envs=num_envs,
                       tensor_args=ctrl_tensor_args,
                       **task_config)
    model.set_task(task)

    # Perform the rollouts
    with torch.no_grad():
        trajectories = ppo_rollout(env=env,
                                   model=model,
                                   tensor_args=ctrl_tensor_args,
                                   save_data=False,
                                   run_critic=False,
                                   **rollout_params)

    # Compute statistics
    success_dict = env.evaluate_success(trajectories)
    stat_dict = utils.compute_statistics(success_dict)

    success_percentage = success_dict['success_percentage']
    mean_success_cost = stat_dict['mean_cost']
    std_success_cost = stat_dict['std_cost']

    print('Success Metric = {:.2f}, Mean Cost = {:.3e} +/- {:.3e}'.format(
        success_percentage,
        mean_success_cost,
        std_success_cost,
    ))

    # Visualize trajectory
    states = trajectories[0]['states']

    mean_state_samples = None
    mppi_state_samples = None

    ref_traj = env.ref_trajectory[0].cpu().T
    env.visualize(states, ref_traj, env.avg_dt)
