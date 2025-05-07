ENV_REGISTRY = {}
import gym
import safety_gym
from torchrl.envs import EnvBase
from envs.safety_gym_envs import SafetyGymEnv
from typing import Optional
import torch
from tensordict import TensorDict
from .integrator import DoubleIntegratorEnv
from typing import Dict, Any, Optional
from copy import deepcopy
from torchrl.envs.transforms import (
    TransformedEnv,
    UnsqueezeTransform,
    CatTensors,
    Compose,
    ObservationNorm,
    DoubleToFloat,
    StepCounter,
)

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
             device: Optional[torch.device] = torch.device("cpu")) -> EnvBase:
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
    return ENV_REGISTRY[env_name](env_name,cfg, device)

@register_env("double_integrator")
def make_double_integrator_env(name:str, cfg: dict, device:Optional[torch.device]) -> EnvBase:
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

    obs_signals = cfg["obs_signals"]
    ref_signals = cfg["ref_signals"]
    base_env = DoubleIntegratorEnv(
        batch_size=cfg.get("num_parallel_env"),
        seed=cfg["seed"],
        td_params=parameters,
        device=device
    )
    transforms = [
            UnsqueezeTransform(in_keys=obs_signals+ref_signals, 
                            dim=-1,
                            in_keys_inv=obs_signals+ref_signals,),
            CatTensors(in_keys=obs_signals, out_key= "observation",del_keys=False,dim=-1),
            CatTensors(in_keys=ref_signals, out_key= "reference",del_keys=False,dim=-1),
            ObservationNorm(in_keys=["observation"], out_keys=["observation"]),
            ObservationNorm(in_keys=["reference"], out_keys=["reference"]),
            CatTensors(in_keys=["observation","reference"], out_key="observation_extended",del_keys=False,dim=-1),
            DoubleToFloat(),
            StepCounter(max_steps=cfg["max_steps"])]
    env = TransformedEnv(
        base_env,
        Compose(
            *transforms
        )
    ).to(device)
    env.transform[3].init_stats(num_iter=1000,reduce_dim=(0,1),cat_dim=1)
    env.transform[4].init_stats(num_iter=1000,reduce_dim=(0,1),cat_dim=1)
    return env

def make_safety_gym_env(env_id: str, cfg: dict, device: Optional[torch.device]=torch.device("cpu")) -> EnvBase:
    """Creates a Safety Gym environment.

    Args:
        env_id (str): The ID of the Safety Gym environment.
        cfg (dict): Configuration dictionary.
        device (torch.device): Device to use.

    Returns:
        EnvBase: The Safety Gym environment.
    """

    
    base_env = SafetyGymEnv(env_id,
            num_envs=cfg.get("num_parallel_env",None))
    base_env.set_seed(cfg["seed"])
    return TransformedEnv(
        base_env,
        StepCounter(max_steps=cfg["max_steps"])
    ).to(device)

for env_id in gym.envs.registry:
    if env_id.startswith("Safexp"):
        # Register all Safety Gym environments
        ENV_REGISTRY[env_id] = make_safety_gym_env
