from torchrl.data import TensorSpec
import wandb
import torch
from math import ceil
from typing import Optional, List
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from matplotlib import pyplot as plt
import math
from datetime import datetime
from torchrl.envs import (
    ExplorationType,
    set_exploration_type,
    step_mdp,
    Compose,
    EnvBase,
    BatchSizeTransform,
    Transform,
    TransformedEnv
)
from utils.utils import reset_batched_env

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
            wandb.log({**logs})
        del eval_rollout
    return logs, eval_str

    
# TODO: Add functionality for multi-step evaluation of Bellman violation?
def calculate_bellman_violation(resolution:int, 
                                value_net:nn.Module,
                                state_space:dict,
                                policy_module:TensorDictModule,
                                base_env:EnvBase,
                                gamma:float,
                                state_group_name:Optional[str] = "obs",
                                after_batch_transform:Optional[List[Transform]] = [],
                                before_batch_transform:Optional[List[Transform]] = []):

    """Calculates the violation of the Bellman equation for the value function
    across the state space. This is useful to check if the value function is
    close to the optimal value function.
    Args:
        resolution (int): The resolution of the state space grid.
        value_net nn.Module: The value function module.
        state_space dict: A dictionary containing a representation of the state space over
            which the value function should be evaluated. The dictionary should contain
            the keys "low" and "high" for each state "state_name"
            which represent the lower and upper bounds of the
            state space respectively. E.g state_space = {"state_name": {"low": -1, "high": 1}}
        policy_module (TensorDictModule): The policy module.
        base_env (EnvBase): The environment without any transformations.
        This is done so that the correct batch transformation can be applied to the
        base environment.
        gamma (float): The discount factor.
        state_group_name (str): The name to group the states under. This is useful when
        the policy module expects the states to be grouped under a certain name.
        if None, the states are not grouped. Default is "obs".
        after_batch_transform (Transform): A transform to apply to the base env
        after the batch transformation. Default is empty list.
        before_batch_transform (Transform): A transform to apply to the base env
        before the batch transformation. Default is empty list.
    """
    dim = len(state_space)
    device = torch.device("cpu")
    # Evaluate on cpu
    value_net.to(device)
    policy_module.to(device)
    linspaces = [torch.linspace(state_space[state_name]["low"],
                                state_space[state_name]["high"],
                                ceil(resolution*(state_space[state_name]["high"]-\
                                    state_space[state_name]["low"])))
                 for state_name in state_space]
    mesh = torch.meshgrid(*linspaces,indexing="xy")
    inputs = torch.stack([m.flatten() for m in mesh],dim=-1)
    batch_size = inputs.shape[0]
    transforms = [*before_batch_transform,
                  BatchSizeTransform(batch_size=[batch_size],
                                     reset_func=reset_batched_env,
                                     env_kwarg=True),
                  *after_batch_transform]
    transforms = [t.clone() for t in transforms if t is not None]
    env = TransformedEnv(
        base_env,
        Compose(*transforms)
    )
    td = env.gen_params(device=device,batch_size=[inputs.shape[0]])
    td = env.reset(td)
    for i, state_name in enumerate(state_space):
        td[state_name] = inputs[...,i]
    if state_group_name is not None:
        td[state_group_name] = torch.stack([td[state_name] for state_name in state_space],
                                          dim=-1)
    values = value_net(inputs).cpu().detach().numpy()
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        td = env.step(policy_module(td))
        td_next = step_mdp(td)
        next_values = value_net(td_next["obs"])
    costs = -td["next","reward"]
    done = td_next["done"]
    bellman_violation_tensor = (costs + gamma*next_values*~done - values).abs()
    bellman_violation_tensor = bellman_violation_tensor.reshape(mesh[0].shape)
    return bellman_violation_tensor