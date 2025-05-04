ENV_REGISTRY = {}
import gym
import safety_gym
from torchrl.envs import EnvBase
from envs.safety_gym_envs import SafetyGymEnv
from typing import Optional
import torch
from tensordict import TensorDict
from .integrator import MultiObjectiveDoubleIntegratorEnv
from typing import Dict, Any, Optional
from copy import deepcopy

def register_env(name):
    def decorator(cls_or_fn):
        ENV_REGISTRY[name] = cls_or_fn
        return cls_or_fn
    return decorator

def registry():
    """Returns the environment ids in the registry.""" 
    return list(ENV_REGISTRY.keys())
def make_env(env_name: str, 
             cfg: Optional[Dict[str,Any]],
             device: Optional[torch.device] = None) -> EnvBase:
    """Creates an environment based on the given name.
    Args:
        env_name (str): The name of the environment to create.
        cfg (dict): Configuration dictionary. 
        device (torch.device): Device to use.
    Returns:
        EnvBase: The created environment.
    Raises:
        ValueError: If the environment name is not recognized.
    """
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env: {env_name}")
    return ENV_REGISTRY[env_name](cfg, device)

@register_env("double_integrator")
def make_double_integrator_env(cfg: dict, device:Optional[torch.device]) -> EnvBase:
    """Creates a double integrator environment.

    Args:
        cfg (dict): Configuration dictionary.
        device (torch.device): Device to use.

    Returns:
        EnvBase: The double integrator environment.
    """
    parameters = TensorDict(
        cfg["params"],
        batch_size=[], device=device
    )
    return MultiObjectiveDoubleIntegratorEnv(
        batch_size=cfg.get("num_parallel_env"),
        seed=cfg["seed"],
        td_params=parameters,
        device=device
    )

for env_id in gym.envs.registry:
    if env_id.startswith("Safexp"):
        # Register all Safety Gym environments
        ENV_REGISTRY[env_id] =(
            lambda cfg, device, env_id=env_id: SafetyGymEnv(env_id, 
                                                        device=device,
                                                        num_envs=cfg.get("num_parallel_env",None))
        )
