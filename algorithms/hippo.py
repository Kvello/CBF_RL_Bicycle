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
from .advantages.multi_GAE import MultiGAE
from torchrl.objectives.value import ValueEstimatorBase
from torchrl.envs.utils import ExplorationType
from .losses.hippo_loss import HiPPOLoss
from tqdm import tqdm
import wandb
from utils.utils import get_config_value
from .ppo import PPO
import warnings
from utils.utils import gradient_projection


class HierarchicalPPO(PPO):
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

        warn_str = "CBF_critic_coef not found in config, using default value of 1.0"
        self.CBF_critic_coef = get_config_value(config, "CBF_critic_coef", 1.0, warn_str)

        warn_str = "secondary_critic_coef not found in config, using default value of 1.0"
        self.secondary_critic_coef = get_config_value(config, "secondary_critic_coef", 1.0, warn_str)

        warn_str = "safety_objective_coef not found in config, using default value of 1.0"
        self.safety_objective_coef = get_config_value(config, "safety_objective_coef", 1.0, warn_str)
        
        warn_str = "secondary_objective_coef not found in config, using default value of 1.0"
        self.secondary_objective_coef = get_config_value(config, "secondary_objective_coef", 1.0, warn_str)
        
        warn_str = "gamma not found in config, using default value of 0.99"
        self.gamma = get_config_value(config, "gamma", 0.99, warn_str)

        warn_str = "num_epochs not found in config, using default value of 10"
        self.num_epochs = get_config_value(config, "num_epochs", 10, warn_str) 
        
        warn_str = "primary_reward_key not found in config, using default value of 'r1'"
        self.primary_reward_key = get_config_value(config, "primary_reward_key", "r1", warn_str)
        
        warn_str = "secondary_reward_key not found in config, using default value of 'r2'"
        self.secondary_reward_key = get_config_value(config, "secondary_reward_key", "r2", warn_str)

        self.loss_value_log_keys = {
            "loss_safety_objective",
            "loss_secondary_objective",
            "loss_CBF",
            "loss_secondary_critic",
        }
        self.reward_keys = {self.primary_reward_key, self.secondary_reward_key}

    def train(self,
              policy_module: TensorDictModule,
              V_primary: TensorDictModule,
              V_secondary: TensorDictModule,
              optim: torch.optim.Optimizer,
              collector: DataCollectorBase,
              replay_buffer: TensorDictReplayBuffer,
              eval_func: Optional[Callable[None, Dict[str, float]]] = None,
              ):
        """Train the PPO algorithm.

        Args:
            policy_module (TensorDictModule): The policy module.
            V_primary (TensorDictModule): The primary value network.
            V_secondary (TensorDictModule): The secondary value network.
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
        if V_primary.out_keys[0] == V_secondary.out_keys[0]:
            warnings.warn("Value networks have the same output keys. This may cause issues.")
            
        if self.config.get("track", False):
            wandb.init(project=self.config.get("wandb_project", "ppo"),
                    sync_tensorboard=True,
                    monitor_gym=True,
                    save_code=True,
                    name=self.config.get("experiment_name", None),
                    config = {**self.config,"method":"hippo"})
            
        self.loss_module = HiPPOLoss(
            actor=policy_module,
            primary_critic=V_primary,
            secondary_critic=V_secondary,
            clip_epsilon=self.clip_epsilon,
            CBF_critic_coef=self.CBF_critic_coef,
            secondary_critic_coef=self.secondary_critic_coef,
            safety_objective_coef=self.safety_objective_coef,
            secondary_objective_coef=self.secondary_objective_coef,
            gamma=self.gamma,
        )
        self.advantage_module = MultiGAE(
            gamma=self.gamma,
            lmdba=self.lmbda,
            V_primary=V_primary,
            V_secondary=V_secondary,
            device=self.device,
            primary_reward_key=self.primary_reward_key,
            secondary_reward_key=self.secondary_reward_key,
        )
        self.optim = optim(
            list(policy_module.parameters()) + 
            list(V_primary.parameters()) +
            list(V_secondary.parameters()),
            **self.config.get("optim_kwargs", {})
        )
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

    def _set_gradients(self,
                      loss_module:TensorDictModule,
                      tensordict: TensorDict):
        """
        Set the gradients for the networks and return the loss values.
        
        Args:
            loss_module (TensorDictModule): The loss module.
            tensordict (TensorDict): The data tensor dictionary.
            
        Returns:
            TensorDict: The loss values.
        """
         
        loss_vals = loss_module(tensordict)
        critic_loss = (
            loss_vals["loss_CBF"] + loss_vals["loss_secondary_critic"]
        )
        critic_loss.backward()
        policy_grad = gradient_projection(loss_module.actor_network, 
                            loss_vals["loss_safety_objective"], 
                            loss_vals["loss_secondary_objective"])
                # Set gradient to the policy module
        last_param_idx = 0
        for p in loss_module.actor_network.parameters():
            new_grad = policy_grad[last_param_idx:last_param_idx + p.data.numel()]
            new_grad = new_grad.view_as(p.data)
            p.grad = new_grad
            last_param_idx += p.data.numel()
        # this is not strictly mandatory but it's good practice to keep
        # your gradient norm bounded
        # torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 
        #                                self.max_grad_norm)
        return loss_vals
