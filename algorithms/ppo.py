from collections import defaultdict
import torch
from typing import List, Dict, Any, Optional, Callable
from tensordict.nn import TensorDictModule, TensorDictModuleBase
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
from torchrl.objectives import ClipPPOLoss, LossModule
from tqdm import tqdm
import wandb
from utils.utils import get_config_value
from .RlAlgoBase import RLAlgoBase
import warnings


class PPO(RLAlgoBase):
    def __init__(self):
        super().__init__()
    def setup(self, config: Dict[str, Any]):
        self.config = config 
        warn_str = "sub_batch_size not found in config, using default value of 256"
        self.sub_batch_size = get_config_value(config, "sub_batch_size", 256,warn_str) 

        warn_str = "max_grad_norm not found in config, using default value of 1.0"
        self.max_grad_norm = get_config_value(config, "max_grad_norm", 1.0)
        
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

        warn_str = "primary_reward_key not found in config, using default value of 'r1'"
        self.primary_reward_key = get_config_value(config, "primary_reward_key", "r1", warn_str)
        
        warn_str = "secondary_reward_key not found in config, using default value of 'r2'"
        self.secondary_reward_key = get_config_value(config, "secondary_reward_key", "r2", warn_str)

        self.loss_value_log_keys = ["loss_safety_objective", "loss_CBF"]
        self.reward_keys = {self.primary_reward_key, self.secondary_reward_key}
    def train(self,
              policy_module: TensorDictModule,
              value_module: TensorDictModule,
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
                    config = {**self.config,"method": "ppo"})
            
        self.advantage_module = GAE(
            gamma=self.gamma,
            lmbda=self.lmbda,
            value_network=value_module,
            average_gae=True,
            device=self.device,
        )
        self.advantage_module.set_keys(
            value = value_module.out_keys[0],
            reward = self.primary_reward_key,
            advantage = "advantage",
            value_target = "value_target",
        )

        self.loss_module = ClipPPOLoss(
            actor_network=policy_module,
            critic_network=value_module,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=False,
            entropy_coef=0.0,
            critic_coef=self.critic_coef,
            loss_critic_type=self.loss_critictype,
        )
        self.loss_module.set_keys(
            advantage="advantage",
            value=value_module.out_keys[0],
            value_target="value_target",
            reward=self.primary_reward_key,
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
                                   self.advantage_module,
                                   self.optim,
                                   replay_buffer,
                                   eval_func=eval_func))
            pbar.update(tensordict_data.numel())
            # scheduler.step()
            if self.config.get("track", False):
                wandb.log({**logs})
            else:
                cum_primary_reward_str = \
                    f"average primary reward={logs[self.primary_reward_key]: 4.4f}"
                cum_secondary_reward_str = \
                    f"average secondary reward={logs[self.secondary_reward_key]: 4.4f}"
                stepcount_str = f"step count (max): {logs['step_count']}"
                lr_str = f"lr policy: {logs['lr']: 4.6f}"
                eval_str = ", ".join([f"{key}: {val}" for key, val in logs.items()])
                pbar.set_description(
                    f"{cum_primary_reward_str},\
                    {cum_secondary_reward_str},\
                    {stepcount_str}, \
                    {lr_str}, \
                    {eval_str}"
                )
        if wandb.run is not None:
            wandb.finish() 
        collector.shutdown()

    def step(self, 
             tensordict_data: TensorDict,
             loss_module: LossModule,
             advantage_module: TensorDictModuleBase,
             optim: torch.optim.Optimizer,
             replay_buffer: TensorDictReplayBuffer,
             eval_func: Optional[Callable[None, Dict[str, float]]] = None):

        """
        Perform a single step of the (general) PPO algorithm.
        
        Args:
            tensordict_data (TensorDict): The data tensor dictionary.
            loss_module (HiPPOLoss): The loss module.
            advantage_module (ValueEstimatorBase): The advantage module.
            optim (torch.optim.Optimizer): The optimizer.
            replay_buffer (TensorDictReplayBuffer): The replay buffer.
            eval_func (Optional[Callable[None, Dict[str, float]]): An optional evaluation function.
            
        Returns:
            Dict[str, float]: The logs.
        """
        if self.config is None:
            raise ValueError("Setup must be called with a config before training")
        logs = defaultdict(list)
        for key in self.loss_value_log_keys:
            logs[key] = 0.0
        for _ in range(self.num_epochs):
            advantage_module(tensordict_data.to(self.device))
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)
            for _ in range(self.frames_per_batch // self.sub_batch_size):
                subdata = replay_buffer.sample(self.sub_batch_size).to(self.device)
                loss_vals = self._set_gradients(loss_module, subdata)
                
                optim.step()
                optim.zero_grad()

                for key in self.loss_value_log_keys:
                    logs[key] += loss_vals[key].item()
        for key in self.loss_value_log_keys:
            logs[key] /= self.num_epochs
        for key in self.reward_keys:
            logs[key] = tensordict_data["next",key].to(torch.float32).mean().item()
        logs["step_count(average)"] = tensordict_data["step_count"].to(torch.float32).mean().item()
        logs["lr"] = optim.param_groups[0]["lr"]
        if eval_func is not None:
            logs.update(eval_func(tensordict_data))
        return logs

    def _set_gradients(self,
        loss_module: ClipPPOLoss,
        tensordict: TensorDict):
        """
        Set the gradients for the loss module and return the loss values.
        
        Args:
            loss_module (ClipPPOLoss): The loss module.
            tensordict (TensorDict): The data tensor dictionary.
            
        Returns:
            Dict[str, float]: The loss values.
        """
        value_key = loss_module.critic_network.out_keys[0]
        state_value = loss_module.critic_network(tensordict)[value_key]
        with torch.no_grad():
            tensordict["td_error"] = torch.abs(tensordict["value_target"] - state_value)
        critic_loss = torch.nn.HuberLoss(reduction='none')(state_value, tensordict["value_target"])
        # Allow for importance sampling of the samples
        if "_weight" in tensordict:
            critic_loss = (tensordict["_weight"] * critic_loss).mean() 
            # I think this is a valid way of weighting the actor loss
            # The tensordict sampled from the replay buffer is a copy of the original,
            # so we can safely modify it without affecting the data in the replay buffer
            tensordict["advantage"] = tensordict["advantage"]* tensordict["_weight"]
        else:
            critic_loss = critic_loss.mean()
        loss_vals = loss_module(tensordict)
        loss_vals["loss_critic"] = critic_loss * self.critic_coef
        # rename loss value keys
        loss_vals["loss_CBF"] = loss_vals["loss_critic"]
        loss_vals["loss_safety_objective"] = loss_vals["loss_objective"]
        del loss_vals["loss_critic"]
        del loss_vals["loss_objective"]
        loss_value = (
            loss_vals["loss_safety_objective"]
            + loss_vals["loss_CBF"]
        )
        
        loss_value.backward()
        return loss_vals