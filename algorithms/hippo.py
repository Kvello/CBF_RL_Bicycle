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
from .losses.hippo_loss import HiPPOLoss
from tqdm import tqdm
import wandb
from utils.utils import get_config_value
from .ppo import PPO
import warnings


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

        warn_str = "primary_critic_coef not found in config, using default value of 1.0"
        self.primary_critic_coef = get_config_value(config, "primary_critic_coef", 1.0, warn_str)

        warn_str = "secondary_critic_coef not found in config, using default value of 1.0"
        self.secondary_critic_coef = get_config_value(config, "secondary_critic_coef", 1.0, warn_str)

        warn_str = "primary_objective_coef not found in config, using default value of 1.0"
        self.primary_objective_coef = get_config_value(config, "primary_objective_coef", 1.0, warn_str)
        
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
                    config = self.config)
            
        self.loss_module = HiPPOLoss(
            actor=policy_module,
            primary_critic=V_primary,
            secondary_critic=V_secondary,
            clip_epsilon=self.clip_epsilon,
            primary_critic_coef=self.primary_critic_coef,
            secondary_critic_coef=self.secondary_critic_coef,
            primary_objective_coef=self.primary_objective_coef,
            secondary_objective_coef=self.secondary_objective_coef,
            gamma=self.gamma,
        )
        self.A_primary = GAE(
            gamma=self.gamma,
            lmbda=self.lmbda,
            value_network=V_primary,
            average_gae=True,
            device=self.device,
        )
        self.A_primary.set_keys(
            reward=self.primary_reward_key,
            advantage="A1",
            value_target="V1_target",
            value=V_primary.out_keys[0]
        )
        self.A_secondary = GAE(
            gamma=self.gamma,
            lmbda=self.lmbda,
            value_network=V_secondary,
            average_gae=True,
            device=self.device,
        )
        self.A_secondary.set_keys(
            reward=self.secondary_reward_key,
            advantage="A2",
            value_target="V2_target",
            value=V_secondary.out_keys[0]
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
                                   self.A_primary,
                                   self.A_secondary,
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
             loss_module: HiPPOLoss,
             A_primary: GAE,
             A_secondary: GAE,
             optim: torch.optim.Optimizer,
             replay_buffer: TensorDictReplayBuffer,
             eval_func: Optional[Callable[None, Dict[str, float]]] = None):

        """
        Perform a single step of the Hierarchical PPO algorithm.
        
        Args:
            tensordict_data (TensorDict): The data tensor dictionary.
            loss_module (HiPPOLoss): The loss module.
            A_primary (GAE): The primary advantage module.
            A_secondary (GAE): The secondary advantage module.
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
            A_primary(tensordict_data.to(self.device))
            A_secondary(tensordict_data.to(self.device))
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)
            for _ in range(self.frames_per_batch // self.sub_batch_size):
                subdata = replay_buffer.sample(self.sub_batch_size).to(self.device)
                loss_vals = self._set_gradients(loss_module, subdata)
                replay_buffer.update_tensordict_priority(subdata) 
                
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

def gradient_projection(
    common_module:torch.nn.Module,
    primary_loss:torch.Tensor,
    secondary_loss:torch.Tensor):
    
        # Primary objective loss gradient
        primary_loss.backward()
        grad_vec_primary_loss = torch.cat(
            [p.grad.view(-1) for p in common_module.parameters()]
        ) 
        common_module.zero_grad()
        # Secondary objective loss gradient
        secondary_loss.backward()
        grad_vec_secondary_loss = torch.cat(
            [p.grad.view(-1) for p in common_module.parameters()]
        )
        common_module.zero_grad()
        # Projection of secondary objective loss gradient onto nullspace of
        # primary objective loss gradient
        secondary_proj =(
            torch.dot(grad_vec_secondary_loss, grad_vec_primary_loss)
            / torch.dot(grad_vec_primary_loss, grad_vec_primary_loss)
            * grad_vec_primary_loss
        )

        grad = (
            grad_vec_secondary_loss - secondary_proj + grad_vec_primary_loss
        )
        return grad