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
        if V_primary.out_keys[0] == V_secondary.out_keys[0]:
            warnings.warn("Value networks have the same output keys. This may cause issues.")
            
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

        # Could probably just write a custom loss function at this point, but 
        # the torchrl implementation is fine for now
        # Need to figure out if I should set separate_losses to True or not
        self.primary_loss = ClipPPOLoss(
            actor_network=policy_module,
            critic_network=V_primary,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=False,
            entropy_coef=0.0,
            critic_coef=self.critic_coef,
            loss_critic_type=self.loss_critictype,
        )
        self.primary_loss.set_keys(
            advantage="A1",
            value=V_primary.out_keys[0],
            value_target="V1_target",
            reward=self.primary_reward_key,
        )
        self.secondary_loss = ClipPPOLoss(
            actor_network=policy_module,
            critic_network=V_secondary,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=False,
            entropy_coef=0.0,
            critic_coef=self.critic_coef,
            loss_critic_type=self.loss_critictype,
        )
        self.secondary_loss.set_keys(
            advantage="A2",
            value=V_secondary.out_keys[0],
            value_target="V2_target",
            reward=self.secondary_reward_key,
        )

        self.optim = optim(
            list(self.primary_loss.parameters()) + list(self.secondary_loss.parameters()),
            **self.config.get("optim_kwargs", {})
        )
        print("Training with config:")
        print(self.config)
        logs = defaultdict(list)
        eval_logs = defaultdict(list)
        pbar = tqdm(total=total_frames)
        for i, tensordict_data in enumerate(collector):
            logs.update(self.step(tensordict_data,
                                   self.primary_loss,
                                   self.secondary_loss,
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
             primary_loss_module: TensorDictModule,
             secondary_loss_module: TensorDictModule,
             A_primary: GAE,
             A_secondary: GAE,
             optim: torch.optim.Optimizer,
             replay_buffer: TensorDictReplayBuffer,
             eval_func: Optional[Callable[None, Dict[str, float]]] = None):

        if self.config is None:
            raise ValueError("Setup must be called with a config before training")
        logs = defaultdict(list)
        logs["loss_primary_objective"] = 0.0
        logs["loss_secondary_objective"] = 0.0
        logs["loss_primary_critic"] = 0.0
        logs["loss_secondary_critic"] = 0.0
        for _ in range(self.num_epochs):
            A_primary(tensordict_data.to(self.device))
            A_secondary(tensordict_data.to(self.device))
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)
            for _ in range(self.frames_per_batch // self.sub_batch_size):
                subdata = replay_buffer.sample(self.sub_batch_size).to(self.device)
                primary_value_key = primary_loss_module.critic_network.out_keys[0]
                secondary_value_key = secondary_loss_module.critic_network.out_keys[0]
                primary_value = primary_loss_module.critic_network(subdata)[primary_value_key]
                secondary_value = secondary_loss_module.critic_network(subdata)[secondary_value_key]

                with torch.no_grad():
                    subdata["td_error"] = torch.abs(subdata["V1_target"] - primary_value)
                    replay_buffer.update_tensordict_priority(subdata)
                primary_critic_loss = torch.nn.HuberLoss(reduction='none')(primary_value, subdata["V1_target"])
                secondary_critic_loss = torch.nn.HuberLoss(reduction='none')(secondary_value, subdata["V2_target"])
                A1 = subdata.get("A1", None) # Store before modifying
                # Allow for importance sampling of the samples weigthed on the primary critic td error
                if "_weight" in subdata:
                    primary_critic_loss = (subdata["_weight"] * critic_loss).mean() 
                    secondary_critic_loss = (subdata["_weight"] * secondary_critic_loss).mean()
                    # I think this is a valid way of weighting the actor loss
                    subdata["A1"] = subdata["A1"]* subdata["_weight"]
                    subdata["A2"] = subdata["A2"]* subdata["_weight"]
                else:
                    primary_critic_loss = primary_critic_loss.mean()
                    secondary_critic_loss = secondary_critic_loss.mean()
                primary_loss_vals = primary_loss_module(subdata)
                secondary_loss_vals = secondary_loss_module(subdata)



                ############################################
                # Policy gradient with projection
                ############################################

                # Primary objective loss gradient
                primary_loss_vals["loss_objective"].backward()
                grad_vec_primary_objective_loss = torch.cat(
                    [p.grad.view(-1) for p in primary_loss_module.actor_net.parameters()]
                ) 
                primary_loss_module.actor_net.zero_grad()
                # Secondary objective loss gradient
                secondary_loss_vals["loss_objective"].backward()
                grad_vec_secondary_objective_loss = torch.cat(
                    [p.grad.view(-1) for p in secondary_loss_module.actor_net.parameters()]
                )
                secondary_loss_module.actor_net.zero_grad()
                # Projection of secondary objective loss gradient onto nullspace of
                # primary objective loss gradient
                secondary_proj =(
                    torch.dot(grad_vec_secondary_objective_loss, grad_vec_primary_objective_loss)
                    / torch.dot(grad_vec_primary_objective_loss, grad_vec_primary_objective_loss)
                    * grad_vec_primary_objective_loss
                )

                policy_grad = (
                    grad_vec_secondary_objective_loss - secondary_proj + grad_vec_primary_objective_loss
                )
                # Set gradient to the policy module
                for p, g in zip(primary_loss_module.actor_net.parameters(), policy_grad):
                    p.grad = g.view_as(p)

                # Critic losses
                primary_loss_vals["loss_critic"] = primary_critic_loss * self.critic_coef
                secondary_loss_vals["loss_critic"] = secondary_critic_loss * self.critic_coef 
                critic_loss = primary_loss_vals["loss_critic"] + secondary_loss_vals["loss_critic"]
                critic_loss.backward()
                 
                
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                # torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 
                #                                self.max_grad_norm)
                optim.step()
                optim.zero_grad()

                logs["loss_primary_objective"] += primary_loss_vals["loss_objective"].item()
                logs["loss_secondary_objective"] += secondary_loss_vals["loss_objective"].item()
                logs["loss_primary_critic"] += primary_critic_loss.item()
                logs["loss_secondary_critic"] += secondary_critic_loss.item()
                # Restore the advantage. I assume the subdata is a reference to the replay
                # buffer data, so we need to restore the original advantage values
                subdata["A1"] = A1
        logs["loss_primary_objective"] /= self.num_epochs
        logs["loss_secondary_objective"] /= self.num_epochs
        logs["loss_primary_critic"] /= self.num_epochs
        logs["loss_secondary_critic"] /= self.num_epochs
        logs[self.primary_reward_key] = tensordict_data["next", self.primary_reward_key].mean().item()
        logs[self.secondary_reward_key] = tensordict_data["next", self.secondary_reward_key].mean().item()
        logs["step_count(average)"] = tensordict_data["step_count"].to(torch.float32).mean().item()
        logs["lr"] = optim.param_groups[0]["lr"]
        if eval_func is not None:
            logs.update(eval_func(tensordict_data))
        return logs