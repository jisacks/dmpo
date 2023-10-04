<h1 align="center">
  DMPO
</h1>
<h2 align="center">
  Deep Model Predictive Optimization
</h2>

<div align="center">
  <a href="https://jisacks.github.io/">Jacob Sacks</a> &nbsp;•&nbsp;
  <a href="https://www.rwik2000.com/">Rwik Rana</a> &nbsp;•&nbsp;
  <a href="https://kevinhuang8.github.io/">Kevin Huang</a> &nbsp;•&nbsp;
  <a href="http://alspitz.github.io/">Alex Spitzer</a> &nbsp;•&nbsp;
  <a href="https://www.gshi.me/">Guanya Shi</a> &nbsp;•&nbsp;
  <a href="https://homes.cs.washington.edu/~bboots/">Byron Boots</a> 
  <br/>
  <br/>
</div>

<p align="center">
    <a href="">Paper</a> |
    <a href="https://sites.google.com/uw.edu/dmpo">Website</a>
</p>

A major challenge in robotics is to design robust policies which enable complex and agile behaviors in the real world. 
On one end of the spectrum, we have model-free reinforcement learning (MFRL), which is incredibly flexible and general 
but often results in brittle policies. In contrast, model predictive control (MPC) continually re-plans at each time 
step to remain robust to perturbations and model inaccuracies. However, despite its real-world successes, MPC often 
under-performs the optimal strategy. This is due to model quality, myopic behavior from short planning horizons, and 
approximations due to computational constraints. And even with a perfect model and enough compute, MPC can get stuck in 
bad local optima, depending heavily on the quality of the optimization algorithm. To this end, we propose Deep Model 
Predictive Optimization (DMPO), which learns the inner-loop of an MPC optimization algorithm directly via experience, 
specifically tailored to the needs of the control problem. We evaluate DMPO on a real quadrotor agile trajectory 
tracking task, on which it improves performance over a baseline MPC algorithm for a given computational budget. 
It can outperform the best MPC algorithm by up to 27% with fewer samples and an end-to-end policy trained with MFRL 
by 19%. Moreover, because DMPO requires fewer samples, it can also achieve these benefits with 4.3X less memory. 
When we subject the quadrotor to turbulent wind fields with an attached drag plate, DMPO can adapt zero-shot while 
still outperforming all baselines.

## Installation
Create a new conda environment with:
```
conda env create -f environment.yml
conda activate dmpo
```
and then install the repository with:
```
cd src
pip install -e .
cd ..
```
## Test that the MPPI baseline works correctly
We provide a test script in ```scripts/run_dmpo_quadrotor.py```. There is a ```is_mppi``` flag in the configuration section
of the script, which if True, will run DMPO in MPPI mode (no learned residual). This is a good sanity check to make sure
things installed properly. 
The script will display the total cost of the trajectory, and enable visualization in the browser via meshcat.
You can run the script by:
```
cd scripts
python run_dmpo_quadrotor.py
```
To test a trained model, turn the ```is_mppi``` off and specify the model location as the ```model_file``` variable.

## How to train a DMPO policy
All experiment configurations are specified via YAML files.
We provide an example for training a quadrotor to perform a zig-zag with yaw flips in 
```config/experiments/quadrotor_dmpo_zigzagyaw.yml```.
To run a training session with this configuration file, perform the following commands:
```
python ppo_main.py --config ../config/experiments/quadrotor_dmpo_zigzagyaw.yml
```
Once the model is done training, provide the correct path to the ```run_dmpo_quadrotor.py``` test script, run an episode,
and visualize with meshcat.

## License
The majority of DMPO is licensed under MIT license, however portions of the project are available under separate license 
terms. Pytorch-Lightning is under the Apache License 2.0 license. 
See [LICENSE](https://github.com/jisacks/dmpo/blob/main/LICENSE) for details.
