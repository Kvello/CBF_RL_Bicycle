from algorithms.ppo import PPO
from models.factory import SafetyValueFunctionFactory
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

import os
import tempfile
from models.policy import ProbabilisticPolicyNet
from ray.train import Checkpoint
from ray.train import report
from tensordict.nn.distributions import NormalParamExtractor
from collections import defaultdict
import torch
from typing import List, Dict, Any
from tensordict.nn import TensorDictModule
from torch import nn
from tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    CatTensors,
    UnsqueezeTransform,
    EnvCreator,
    BatchSizeTransform
)
from torchrl.envs.utils import ExplorationType
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from envs.integrator import SafeDoubleIntegratorEnv, plot_integrator_trajectories, plot_value_function_integrator
from datetime import datetime
import argparse
from results.evaluate import evaluate_policy, calculate_bellman_violation
from utils.utils import reset_batched_env   
import wandb
from models.factory import SafetyValueFunctionFactory
from algorithms.ppo import PPO

# TODO: Implement
def create_policy_net_config(config):
    policy_net_config = {
        "input_size": config["input_size"],
        "action_size": config["action_size"],
        "layers": [config["net_layer_size"] for _ in range(config["net_layers"])],
        "device": config["device"],
        "activation": getattr(nn, config["net_activation"])(),
    }
    return policy_net_config
def create_value_net_config(config):
    value_net_config = {
        "name": config["value_net_architecture"],
        "input_size": config["input_size"],
        "layers": [config["net_layer_size"] for _ in range(config["net_layers"])],
        "device": config["device"],
        "activation": getattr(nn, config["net_activation"])(),
        "bounded": config["bounded_value"],
        "eps": config["eps"],
    }
    return value_net_config
def tune_ppo(config):
    """Tune the PPO algorithm.
    
    Args:
        config (Dict): The configuration dictionary.
    """
    ###################
    # Environment setup
    ###################
    state_space = {"x1": {"low": -config["max_x1"], "high": config["max_x1"]},
                    "x2": {"low": -config["max_x2"], "high": config["max_x2"]}}

    max_rollout_len = config.get("max_rollout_len")
    parameters = TensorDict({
        "params" : TensorDict({
            "dt": config["dt"],
            "max_x1": config["max_x1"],
            "max_x2": config["max_x2"],
            "max_input": config["max_input"],
        },[],
        device=config["device"])
        },[],device=config["device"])
    after_batch_transform = [
            UnsqueezeTransform(in_keys=["x1", "x2"], dim=-1,in_keys_inv=["x1","x2"]),
            CatTensors(in_keys =["x1", "x2"], out_key= "obs",del_keys=False,dim=-1),
            ObservationNorm(in_keys=["obs"], out_keys=["obs"]),
            DoubleToFloat(),
            StepCounter(max_steps=max_rollout_len)]
    base_env = SafeDoubleIntegratorEnv(device=device,td_params=parameters)
    env = TransformedEnv(
        base_env,
        Compose(
            BatchSizeTransform(batch_size=[config["batches_per_process"]],
                               reset_func=reset_batched_env,
                               env_kwarg=True),
            *after_batch_transform
        )
    ).to(config["device"])

    env.transform[3].init_stats(num_iter=1000,reduce_dim=(0,1),cat_dim=1)
    ppo_instance = PPO()
    ppo_instance.setup(config)

    action_high = (env.action_spec_unbatched.high if hasattr(env, "action_spec_unbatched") 
                                                    else env.action_spec.high)
    action_low = (env.action_spec_unbatched.low if hasattr(env, "action_spec_unbatched") 
                                                    else env.action_spec.low)
    
    ###################
    # Model setup
    ###################
    value_net_config = create_value_net_config(config)
    value_net = SafetyValueFunctionFactory.create(**value_net_config)
    value_module = ValueOperator(
        module=value_net,
        in_keys=["obs"],
    )
    policy_net_config = create_policy_net_config(config)
    policy_net = ProbabilisticPolicyNet(**policy_net_config)
    
    policy_module = ProbabilisticActor(
        module=TensorDictModule(module=policy_net,
                                in_keys=["obs"],
                                out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"],
        spec=env.action_spec,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_low,
            "high": action_high,
        },
        return_log_prob=True,
    )
    ###################
    # Evaluation function
    ###################
    def ray_eval_func(data):
        logs = defaultdict(list)
        bm_viol = calculate_bellman_violation(
            10,
            value_net,
            state_space, 
            policy_module,
            base_env, 
            config["gamma"],
            after_batch_transform=after_batch_transform
        )
        # Return the mean, max, or std of the Bellman violation?
        logs["bellman_violation_mean"] = bm_viol.flatten().mean().item()
        logs["bellman_violation_max"] = bm_viol.flatten().max().item()
        logs["bellman_violation_std"] = bm_viol.flatten().std().item()
        return logs
    ###################
    # Data collection
    ###################
    env_creator = EnvCreator(lambda: env)
    collector = SyncDataCollector(
        create_env_fn=env_creator,
        policy=policy_module,
        frames_per_batch=config["frames_per_batch"],
        total_frames=config.get("frames_per_search", int(2**10)),
        split_trajs=False,
        device=config["device"],
        exploration_type=ExplorationType.RANDOM)

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=config["frames_per_batch"],device=config["device"]),
        sampler=SamplerWithoutReplacement(),
    )
    optim = torch.optim.Adam
    for i, td_data in enumerate(collector):
        logs = ppo_instance.step(td_data,
                config["frames_per_batch"],
                 value_module, 
                 policy_module, 
                 optim, 
                 replay_buffer, 
                 eval_func=ray_eval_func)
        report({"bellman_violation_std":logs["bellman_violation_std"]})
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(
                {
                    "policy_module": policy_module.state_dict(),
                    "value_module": value_module.state_dict(),
                },
                os.path.join(tmpdir, "checkpoint.pth"),
            )
            checkpoint = Checkpoint.from_directory(tmpdir)
    

if __name__ == "__main__":
    

    device = (
        torch.device(0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    fixed_config = {
        "input_size": 2,
        "action_size": 1,
        "entropy_eps": 0.0,
        "loss_critictype": "smooth_l1",
        "gamma": 0.99,
        "max_x1": 1.0,
        "max_x2": 1.0,
        "max_input": 1.0,
        "dt": 0.05,
        "device": device,
        "frames_per_search": 2**18,
    }
    def wrapped_tune(config):
        config.update(fixed_config)
        config["batches_per_process"] = (
            max(int(2**12),config["frames_per_batch"]//config["max_rollout_len"])
        )
        config["optim_kwargs"] = {"lr": config["lr"]}
        del config["lr"]
        tune_ppo(config)
    search_space = {
        "frames_per_batch": tune.randint(int(2**12), int(2**18)),
        "clip_epsilon": tune.uniform(0.05, 0.3),
        "critic_coef": tune.uniform(0.1, 3.0),
        "lmbda": tune.uniform(0.1, 0.9),
        "max_rollout_len": tune.randint(8, 256),
        "sub_batch_size": tune.randint(128, int(2**12)),
        "max_grad_norm": tune.uniform(0.5, 5.0),
        "lr": tune.loguniform(1e-5, 1e-1),
        "value_net_architecture": tune.choice(["feedforward", "quadratic"]),
        "net_layers": tune.randint(1, 5),
        "net_layer_size": tune.choice([32, 64, 128]),
        "net_activation": tune.choice(["ReLU", "Tanh","ELU","LeakyReLU"]),
        "bounded_value": tune.choice([True, False]),
        "eps": tune.uniform(1e-4, 1e-1),
        "num_epochs": tune.randint(2, 64),
    }
    scheduler = ASHAScheduler(
        metric="bellman_violation_std",
        mode="min",
        max_t=100,
        grace_period=5,
        reduction_factor=2,
    )
    hyperopt_search = HyperOptSearch(
        search_space,
        metric="bellman_violation_std",
        mode="min",
    )
    tuner = tune.Tuner(
        wrapped_tune,
        tune_config=tune.TuneConfig(
            num_samples=20,
            search_alg=hyperopt_search,
            scheduler=scheduler,
        )
    )
    if device.type == "cuda":
        results = tune.run(
            wrapped_tune,
            config=search_space,
            search_alg=hyperopt_search,
            scheduler=scheduler,
            resources_per_trial={"gpu": 1},
        )
    else:
        results = tune.run(
            wrapped_tune,
            search_alg=hyperopt_search,
            scheduler=scheduler,
        )