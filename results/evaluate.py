from torchrl.data import TensorSpec
import wandb
import torch
from math import ceil
from typing import Optional, List, Tuple, Dict, Any, Callable
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

class PolicyEvaluator:
    def __init__(self, 
                 env:EnvBase, 
                 policy_module:TensorDictModule,
                 keys_to_log:List[str] = ["reward","step_count"],
                 rollout_len:int = 1000):
        self.env = env
        self.policy_module = policy_module
        self.keys_to_log = keys_to_log
        self.rollout_len = rollout_len
    def evaluate_policy(self) -> Dict[str,Any]:
        logs = {}
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = self.env.rollout(
                self.rollout_len, self.policy_module,break_when_any_done=False
            )
            for key in self.keys_to_log:
                if key in eval_rollout["next"]:
                    if key == "step_count":
                        logs[f"eval {key}(average)"] = (
                            eval_rollout[key].max(dim=1).values.to(torch.float32).mean().item()
                        )
                    else:
                        logs[f"eval {key}(average)"] = (
                            eval_rollout["next", key].to(torch.float32).mean().item()
                        )
                elif key in eval_rollout:
                    logs[f"eval {key}(average)"] = (eval_rollout[key].to(torch.float32).mean().item())
            del eval_rollout
        return logs

    
# TODO: Add functionality for multi-step evaluation of Bellman violation?
def calculate_bellman_violation(resolution:int, 
                                value_module:TensorDictModule,
                                state_space:dict,
                                policy_module:TensorDictModule,
                                base_env_creator:Callable[...,EnvBase],
                                gamma:float,
                                transforms:Optional[List[Transform]] = []):

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
    Returns:
        bellman_violation_tensor (np.ndarray): The Bellman violation tensor.
        mesh (np.ndarray): The meshgrid of the state space.
    """
    dim = len(state_space)
    device = value_module.device
    value_key = value_module.out_keys[0]
    linspaces = [torch.linspace(state_space[state_name]["low"],
                                state_space[state_name]["high"],
                                ceil(resolution*(state_space[state_name]["high"]-\
                                    state_space[state_name]["low"])))
                 for state_name in state_space]
    mesh = torch.meshgrid(*linspaces,indexing="xy")
    inputs = torch.stack([m.flatten() for m in mesh],dim=-1).to(device)
    batch_size = inputs.shape[0]
    transforms = [t.clone() for t in transforms if t is not None]
    base_env = base_env_creator(batch_size=batch_size,device=device)
    env = TransformedEnv(
        base_env,
        Compose(*transforms)
    )
    def apply_transforms(td:TensorDict):
        for t in transforms:
            td = t(td)
        return td
    td = base_env.reset()
    for i, state_name in enumerate(state_space):
        td[state_name] = inputs[...,i]
    td = apply_transforms(td)
    values = value_module(td)[value_key].detach()
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        td = env.step(policy_module(td))
        td_next = step_mdp(td)
        next_values = value_module(td_next)[value_key].detach()
    rewards = td["next","reward"]
    done = td_next["done"]
    bellman_violation_tensor = (rewards + gamma*next_values*~done - values).abs()
    bellman_violation_tensor = bellman_violation_tensor.reshape(mesh[0].shape)
    mesh_np = [m.cpu().numpy() for m in mesh]
    return bellman_violation_tensor.cpu().numpy(), mesh_np