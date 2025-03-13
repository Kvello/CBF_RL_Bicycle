from collections import defaultdict
import torch
from typing import List, Dict, Any, Optional, Callable
from tensordict.nn import TensorDictModule
from torchrl.collectors import DataCollectorBase
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs import (
    TransformedEnv,
    EnvBase,
    Transform,
)
from torchrl.objectives.value import GAE
from torchrl.envs.utils import ExplorationType
from torchrl.objectives import ClipPPOLoss
from tqdm import tqdm
from results.evaluate import evaluate_policy, calculate_bellman_violation
import wandb
from utils.utils import get_config_value


def train_ppo(
    env: EnvBase,
    gamma: float,
    policy_module: TensorDictModule,
    value_module: TensorDictModule,
    collector: DataCollectorBase,
    replay_buffer: ReplayBuffer,
    optim: torch.optim.Optimizer,
    config: Dict[str, Any],
    eval_func: Optional[Callable[TensorDictModule, Dict[str,Any]]] = None,
    state_space: Optional[Dict[str, Dict[str, float]]] = None,
    before_batch_transform: Optional[List[Transform]] = [],
    after_batch_transform: Optional[List[Transform]] = [],
):
    if config.get("track", False):
        wandb.init(project=config.get("wandb_project", "ppo"),
                   sync_tensorboard=True,
                   monitor_gym=True,
                   save_code=True,
                   name=config.get("experiment_name", None))
        
    num_epochs = config.get("num_epochs", 10)
    if hasattr(collector, "requested_frames_per_batch"):
        frames_per_batch = collector.requested_frames_per_batch
    else:
        raise ValueError("Collector must have requested_frames_per_batch attribute.\
                         Try using a different collector.")
    if hasattr(collector, "total_frames"):
        total_frames = collector.total_frames
    else:
        raise ValueError("Collector must have total_frames attribute.\
                            Try using a different collector.")

    warn_str = "sub_batch_size not found in config, using default value of 256"
    sub_batch_size = get_config_value(config, "sub_batch_size", 256,warn_str) 

    warn_str = "max_grad_norm not found in config, using default value of 1.0"
    max_grad_norm = get_config_value(config, "max_grad_norm", 1.0)
    
    warn_str = "entropy_eps not found in config, using default value of 0.0"
    entropy_eps = get_config_value(config, "entropy_eps", 0.0, warn_str)

    warn_str = "device not found in config, using default value of 'cpu'"
    device = get_config_value(config, "device", "cpu", warn_str)

    warn_str = "clip_epsilon not found in config, using default value of 0.2"
    clip_epsilon = get_config_value(config, "clip_epsilon", 0.2, warn_str)
    
    warn_str = "lmbda not found in config, using default value of 0.95"
    lmbda = get_config_value(config, "lmbda", 0.95, warn_str)

    warn_str = "critic_coef not found in config, using default value of 1.0"
    critic_coef = get_config_value(config, "critic_coef", 1.0, warn_str)

    warn_str = "loss_critic_type not found in config, using default value of 'smooth_l1'"
    loss_critictype = get_config_value(config, "loss_critic_type", "smooth_l1", warn_str)

    print("Training with config:")
    print(config)
    logs = defaultdict(list)
    eval_logs = defaultdict(list)
    pbar = tqdm(total=total_frames)

    advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=value_module,
        average_gae=True,
        device=device,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        # these keys match by default but we set this for completeness
        critic_coef=critic_coef,
        loss_critic_type=loss_critictype,
    )
    optim = optim(loss_module.parameters(), **config.get("optim_kwargs", {}))
    for i, tensordict_data in enumerate(collector):
        logs["loss_objective"] = 0.0
        logs["loss_critic"] = 0.0
        logs["loss_entropy"] = 0.0
        for _ in range(num_epochs):
            advantage_module(tensordict_data.to(device))
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size).to(device)
                loss_vals = loss_module(subdata)
                if entropy_eps == 0.0:
                    loss_vals["loss_entropy"] = torch.tensor(0.0).to(device)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                logs["loss_objective"] += loss_vals["loss_objective"].item()
                logs["loss_critic"] += loss_vals["loss_critic"].item()
                logs["loss_entropy"] += loss_vals["loss_entropy"].item()
                
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()
        logs["loss_objective"] /= num_epochs
        logs["loss_critic"] /= num_epochs
        logs["loss_entropy"] /= num_epochs
        logs["reward"] = tensordict_data["next", "reward"].mean().item()
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward']: 4.4f}"
        )
        logs["step_count"] = tensordict_data["step_count"].max().item()
        stepcount_str = f"step count (max): {logs['step_count']}"
        logs["lr"] = optim.param_groups[0]["lr"]
        lr_str = f"lr policy: {logs['lr']: 4.6f}"
        if config.get("track_bellman_violation",False):
            bellman_eval_res = config.get("bellman_eval_res", 10)
            value_net = value_module.module
            assert value_net is not None, "Value network must be a module"
            assert state_space is not None, "State space must be provided if bellman\
                violation is to be tracked"
            base_env = env.base_env if isinstance(env, TransformedEnv) else env
            bm_viol = calculate_bellman_violation(bellman_eval_res, 
                                                value_net,
                                                state_space, 
                                                policy_module,
                                                base_env, 
                                                gamma,
                                                after_batch_transform=after_batch_transform,
                                                before_batch_transform=before_batch_transform)
                                        
                                                            
            logs["bellman_violation_mean"] = bm_viol.flatten().mean().item()
            logs["bellman_violation_max"] = bm_viol.flatten().max().item()
            logs["bellman_violation_std"] = bm_viol.flatten().std().item()
        if i % 10 == 0 and eval_func is not None:
            eval_logs = eval_func(policy_module)
            for key, val in eval_logs.items():
                logs[key] = val
        if config.get("track", False):
            wandb.log({**logs})
        else:
            eval_str = ", ".join([f"{key}: {val}" for key, val in logs.items()])
            pbar.set_description(
                f"{cum_reward_str}, {stepcount_str}, {lr_str}, {eval_str}"
            )
        # scheduler.step()
    if wandb.run is not None:
        wandb.finish() 
    collector.shutdown()