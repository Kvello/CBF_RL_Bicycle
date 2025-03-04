from torchrl.data import TensorSpec
from torchrl.envs import EnvBase, ExplorationType, set_exploration_type
import wandb
import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from matplotlib import pyplot as plt
import math
from datetime import datetime
def evaluate_policy(env:EnvBase, 
                    policy_module:TensorDictModule,
                    rollout_len:int, 
                    log_wandb: bool = False):
    logs = {}
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        # execute a rollout with the trained policy

        eval_rollout = env.rollout(rollout_len, policy_module)

        logs["eval reward"] = (eval_rollout["next", "reward"].mean().item())
        logs["eval reward (sum)"] = eval_rollout["next", "reward"].sum().item()
        logs["eval step_count"] = (eval_rollout["step_count"].max().item())
        eval_str = (
            f"eval cumulative reward: {logs['eval reward (sum)']: 4.4f} "
            f"(init: {logs['eval reward (sum)']: 4.4f}), "
            f"eval step-count: {logs['eval step_count']}"
        )
        if log_wandb:
            wandb.log(
                {
                    "eval reward": logs["eval reward"],
                    "eval reward (sum)": logs["eval reward (sum)"],
                    "eval step_count": logs["eval step_count"],
                }
            )
        del eval_rollout
    return logs, eval_str

    
def calculate_bellman_violation(state_space:dict,
                                resolution:int, 
                                value_net:nn.Module,
                                policy_net:nn.Module,
                                gamma:float):

    """Calculates the violation of the Bellman equation for the value function
    across the state space. This is useful to check if the value function is
    close to the optimal value function.
    Args:
        state_space dict: A dictionary containing a representation of the state space over
            which the value function should be evaluated. The dictionary should contain
            the keys "low" and "high" for each state "state_name"
            which represent the lower and upper bounds of the
            state space respectively. E.g state_space = {"state_name": {"low": -1, "high": 1}}
        value_module nn.Module: The value function module.
        gamma (float): The discount factor.
    """
    dim = len(state_space)
    linspaces = [torch.linspace(state_space[state_name]["low"],
                                state_space[state_name]["high"],
                                resolution*\
    (state_space[state_name]["high"]-state_space[state_name]["low"])) for state_name in state_space]
    mesh = torch.meshgrid(*linspaces,indexing="xy")
    inputs = torch.stack([m.flatten() for m in mesh],dim=-1)
    outputs = value_net(inputs)
