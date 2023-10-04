import yaml
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Config file to load.')
    args = parser.parse_args()

    # Load the configuration file
    config_file = args.config
    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

    # Set the seed
    seed = config.get('seed', 0)

    # Create the experiment
    from dmpo.experiment.ppo_experiment import PPOExperiment
    exp = PPOExperiment(**config)

    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)

    exp.run()