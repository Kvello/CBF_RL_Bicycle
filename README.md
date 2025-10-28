# Safety Prioritizing Policy Optimization via DTBFs
## :rocket: Overview & Motivation

Real-world RL applications require safety guarantees. S3PO addresses this by:

* **Decoupling** safety and performance into two MDPs.

* **Learning** separate value functions and projecting policy updates to respect safety constraints.

* **Certifying** safety via a Discrete-Time Barrier Function.

This dual-MDP approach enables provably safe learning while obtaining reasonable performance.
## :zap: Quick Start
```bash
# Clone repo
git clone --recurse-submodules git@github.com:Kvello/Certified-Safe-RL-using-DTBFs.git
cd Certified-Safe-RL-using-DTBFs
# Set up Python environment
pip install -r requirements.txt
# Run training (double integrator)
python runner.py train=True config_file=hippo_double_integrator.yaml
```

This repo contains the code used to develop, implement and test an algorithm for safe RL extending PPO. It decomposes the CMDP into two MDPs - one concerned with safety, and one concerned with performance - for which two separate value functions are learned. The policy gradient is updated using a projection algorithm. We call the resulting algorithm S3PO (Safety Prioritizing Policy Optimization). Note that the algorithm was initially named HiPPO, so the namings used in code still use this name. It can be shown formally that the value function associated with the safety preserving MDP is a Discrete Time Barrier Funciton, certifying safety of the learned policy after theoretical convergence.
## :file_folder: Directory Structure
```bash
├── algorithms/           # S3PO implementation (algorithms/hippo.py)
├── envs/                 # Task & environment definitions
├── configs/              # YAML configs for training & eval
├── requirements.txt      # Core deps (Python 3.10.16)
├── requirements_safetygym.txt  # For Safety Gym (Python 3.9.x)
├── runner.py             # Entry point for train/eval
├── models                # Model definitions
├── utils/utils.py        # Utility functions
├── tests                 # tests
└── README.md             # This file
```
## :bulb: Main contributions
The main contributions of this work is the S3PO algorithm found under
```bash
algorithms/hippo.py
```
and the associated loss found in
```bash
algorithms/losses/hippo_loss.py
```
## :hammer_and_wrench: Setup and dependencies
This project depends on a working [Safe Control Gym](https://github.com/Kvello/safe-control-gym) installation, which is included as a submodule. To test on [Safety Gym](https://github.com/Kvello/safety-gym) environments, also a working [mujoco-py](https://github.com/openai/mujoco-py) installation is required. The [MuJoCo](https://mujoco.org/) binaries are included in the submodule. Follow the guide at [mujoco-py](https://github.com/openai/mujoco-py) for setting this up.

The requirements are listed in the requirements.txt file. Simply run
```bash
pip install -r requirements.txt
```
preferably in a clean python environment. Python 3.10.16 was used.
[Pyenv](https://github.com/pyenv/pyenv) or [nix](https://nixos.org/) is highly recommended for setting up the environment.
Optionally any other (python) environment manager can be used.
Unfortunately [Safety Gym](https://github.com/Kvello/safety-gym) requires older package versions than [Safe Control Gym](https://github.com/Kvello/safe-control-gym). It also requires an older python version (we used 3.9.21). The requirements for using [Safety Gym](https://github.com/Kvello/safety-gym) are listed in the requirements_safetygym.txt file. 
In the respective python environment install [Safety Gym](https://github.com/Kvello/safety-gym) OR [Safe Control Gym](https://github.com/Kvello/safe-control-gym) in editable mode by navigating to the correct directory and running
```bash
pip install -e .
```
To run the training simply run.
```bash
python runner.py train=True config_file=<config file name>
```
The config files can be found in
```bash
configs/
```
Results can be visualized by setting 'visualize=True', the determinisic policy will be evaluated during and after training by setting 'eval=True'. See
```bash
config/default.yaml
```
for all the configuration options.
## :gear: Configuration Options
All hyperparameters and flags live in separate yaml files in ```configs/```. Key S3PO fields:
```yaml
num_epochs: 10          # number of opt. epochs per batch
frames_per_batch: 8192  # Simulations steps(transitions) per batch
sub_batch_size: 256     # size of sub-batches
total_frames: 16777216  # Total frames to train for
clip_epsilon: 0.2       # epsillon parameter in PPO loss
lmbda1: 0.95            # lambda parameter for GAE for primary/safety objective
lmbda2: 0.95            # lambda parameter for GAE for secondary/performance objective
critic_coef: 0.5        # weight on critic loss
entropy_coef: 0.01      # entropy coefficient beta
optim_kwargs:
  lr: 1.0e-4            # (initial) learning rate
```
## :chart_with_upwards_trend: W&B Logging & Local Plots
* W&B logging can be enabled by adding wandb=True to the runner command

## :bookmark_tabs: Cite This Work
```latex
@mastersthesis{Kvello2025,
  author    = {Markus Kvello},
  title     = {Safety Prioritizing Policy Optimization via Discrete-Time Barrier Functions},
  school    = {Norwegian University of Science and Technology},
  year      = {2025},
  url       = {https://github.com/Kvello/safe-rl-s3po},
}
```
## :pray: Acknowledgments
* Safe Control Gym and Safety Gym projects
* The [torchrl](https://docs.pytorch.org/rl/stable/index.html) project
* Supervisors:
    1. Professor Dr. Konstantinos Alexis
    2. PhD Candidate Marvin Chayton Harms
