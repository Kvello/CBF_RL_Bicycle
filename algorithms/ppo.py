from collections import defaultdict
import torch
from typing import List, Dict, Any, Optional, Callable
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from torchrl.collectors import DataCollectorBase
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.envs import (
    TransformedEnv,
    EnvBase,
    Transform,
)
from torchrl.objectives.value import GAE
from torchrl.envs.utils import ExplorationType
from torchrl.objectives import ClipPPOLoss
from tqdm import tqdm
import wandb
from utils.utils import get_config_value
from .RlAlgoBase import RLAlgoBase
import warnings


class HierarchicalPPO(RLAlgoBase):
    def __init__(self):
        super().__init__()
    def setup(self, config: Dict[str, Any]):
        self.config = config 
        warn_str = "sub_batch_size not found in config, using default value of 256"
        self.sub_batch_size = get_config_value(config, "sub_batch_size", 256,warn_str) 

        warn_str = "max_grad_norm not found in config, using default value of 1.0"
        self.max_grad_norm = get_config_value(config, "max_grad_norm", 1.0)
        
        warn_str = "entropy_eps not found in config, using default value of 0.0"
        self.entropy_eps = get_config_value(config, "entropy_eps", 0.0, warn_str)

        warn_str = "device not found in config, using default value of 'cpu'"
        self.device = get_config_value(config, "device", "cpu", warn_str)

        warn_str = "clip_epsilon not found in config, using default value of 0.2"
        self.clip_epsilon = get_config_value(config, "clip_epsilon", 0.2, warn_str)
        
        warn_str = "lmbda not found in config, using default value of 0.95"
        self.lmbda = get_config_value(config, "lmbda", 0.95, warn_str)

        warn_str = "critic_coef not found in config, using default value of 1.0"
        self.critic_coef = get_config_value(config, "critic_coef", 1.0, warn_str)

        warn_str = "loss_critic_type not found in config, using default value of 'smooth_l1'"
        self.loss_critictype = get_config_value(config, "loss_critic_type", "smooth_l1", warn_str)

        warn_str = "gamma not found in config, using default value of 0.99"
        self.gamma = get_config_value(config, "gamma", 0.99, warn_str)

        warn_str = "num_epochs not found in config, using default value of 10"
        self.num_epochs = get_config_value(config, "num_epochs", 10, warn_str) 

    def train(self,
              policy_module: TensorDictModule,
              V_primary: TensorDictModule,
              V_secondary: TensorDictModule,
              optim: torch.optim.Optimizer,
              collector: DataCollectorBase,
              replay_buffer: TensorDictReplayBuffer,
              eval_func: Optional[Callable[None, Dict[str, float]]] = None
              ):
        """Train the PPO algorithm.

        Args:
            policy_module (TensorDictModule): The policy module.
            value_module (TensorDictModule): The value module.
            optim (torch.optim.Optimizer): The optimizer.
            collector (DataCollectorBase): The data collector.
            replay_buffer (TensorDictReplayBuffer): The replay buffer.
            An optional evaluation function that evaluates the policy in the environment
            This should be a function that takes in the policy and returns a dict of
            floats of evaluation metrics. Defaults to None.

        Raises:
            ValueError: if the collector does not have the requested_frames_per_batch attribute.
            ValueError: if the collector does not have the total_frames attribute.
        """

        if self.config is None:
            raise ValueError("Setup must be called with a config before training")
        
        if hasattr(collector, "requested_frames_per_batch"):
            self.frames_per_batch = collector.requested_frames_per_batch
        else:
            raise ValueError("Collector must have requested_frames_per_batch attribute.\
                            Try using a different collector.")
        if hasattr(collector, "total_frames"):
            total_frames = collector.total_frames
        else:
            raise ValueError("Collector must have total_frames attribute.\
                                Try using a different collector.")

        if self.config.get("track", False):
            wandb.init(project=self.config.get("wandb_project", "ppo"),
                    sync_tensorboard=True,
                    monitor_gym=True,
                    save_code=True,
                    name=self.config.get("experiment_name", None),
                    config = self.config)
            
        self.A_primary = GAE(
            gamma=self.gamma,
            lmbda=self.lmbda,
            value_network=V_primary,
            average_gae=True,
            device=self.device,
        )
        self.A_secondary = GAE(
            gamma=self.gamma,
            lmbda=self.lmbda,
            value_network=V_secondary,
            average_gae=True,
            device=self.device,
        )

        self.primary_loss = ClipPPOLoss(
            actor_network=policy_module,
            critic_network=value_module,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=bool(self.entropy_eps),
            entropy_coef=self.entropy_eps,
            critic_coef=self.critic_coef,
            loss_critic_type=self.loss_critictype,
        )
        self.secondary_loss = ClipPPOLoss(
            actor_network=policy_module,
            critic_network=value_module,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=bool(self.entropy_eps),
            entropy_coef=self.entropy_eps,
            critic_coef=self.critic_coef,
            loss_critic_type=self.loss_critictype,
        )

        self.optim = optim(self.loss_module.parameters(), **self.config.get("optim_kwargs", {}))

        print("Training with config:")
        print(self.config)
        logs = defaultdict(list)
        eval_logs = defaultdict(list)
        pbar = tqdm(total=total_frames)
        for i, tensordict_data in enumerate(collector):
            logs.update(self.step(tensordict_data,
                                   self.loss_module,
                                   self.optim,
                                   self.advantage_module,
                                   replay_buffer,
                                   eval_func=eval_func))
            pbar.update(tensordict_data.numel())
            # scheduler.step()
            if self.config.get("track", False):
                wandb.log({**logs})
            else:
                cum_reward_str = f"average reward={logs['reward']: 4.4f}"
                stepcount_str = f"step count (max): {logs['step_count']}"
                lr_str = f"lr policy: {logs['lr']: 4.6f}"
                eval_str = ", ".join([f"{key}: {val}" for key, val in logs.items()])
                pbar.set_description(
                    f"{cum_reward_str}, {stepcount_str}, {lr_str}, {eval_str}"
                )
        if wandb.run is not None:
            wandb.finish() 
        collector.shutdown()

    def step(self, 
             tensordict_data: TensorDict,
             loss_module: TensorDictModule,
             optim: torch.optim.Optimizer,
             advantage_module: TensorDictModule,
             replay_buffer: TensorDictReplayBuffer,
             eval_func: Optional[Callable[None, Dict[str, float]]] = None):

        if self.config is None:
            raise ValueError("Setup must be called with a config before training")
        logs = defaultdict(list)
        logs["loss_objective"] = 0.0
        logs["loss_critic"] = 0.0
        logs["loss_entropy"] = 0.0
        for _ in range(self.num_epochs):
            advantage_module(tensordict_data.to(self.device))
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)
            for _ in range(self.frames_per_batch // self.sub_batch_size):
                subdata = replay_buffer.sample(self.sub_batch_size).to(self.device)
                state_value = loss_module.critic_network(subdata)["state_value"]
                with torch.no_grad():
                    subdata["td_error"] = torch.abs(subdata["value_target"] - state_value)
                    replay_buffer.update_tensordict_priority(subdata)
                critic_loss = torch.nn.HuberLoss(reduction='none')(state_value, subdata["value_target"])
                advantage = subdata.get("advantage", None) # Store before modifying
                # Allow for importance sampling of the samples
                if "_weight" in subdata:
                    critic_loss = (subdata["_weight"] * critic_loss).mean() 
                    # I think this is a valid way of weighting the actor loss
                    subdata["advantage"] = subdata["advantage"]* subdata["_weight"]
                else:
                    critic_loss = critic_loss.mean()
                loss_vals = loss_module(subdata)
                loss_vals["loss_critic"] = critic_loss * self.critic_coef
                if self.entropy_eps == 0.0:
                    loss_vals["loss_entropy"] = torch.tensor(0.0).to(self.device)
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
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 
                                               self.max_grad_norm)
                optim.step()
                optim.zero_grad()
                # Restore the advantage. I assume the subdata is a reference to the replay
                # buffer data, so we need to restore the original advantage values
                subdata["advantage"] = advantage
        logs["loss_objective"] /= self.num_epochs
        logs["loss_critic"] /= self.num_epochs
        logs["loss_entropy"] /= self.num_epochs
        logs["reward"] = tensordict_data["next", "reward"].mean().item()
        logs["step_count(average)"] = tensordict_data["step_count"].to(torch.float32).mean().item()
        logs["lr"] = optim.param_groups[0]["lr"]
        if eval_func is not None:
            logs.update(eval_func(tensordict_data))
        return logs