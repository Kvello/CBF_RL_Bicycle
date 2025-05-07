from collections import defaultdict
import torch
from typing import List, Dict, Any, Optional, Callable
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from tensordict import TensorDict
from torchrl.collectors import DataCollectorBase
from torchrl.data import ReplayBuffer, RandomSampler, LazyTensorStorage
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

        warn_str = "entropy_coef not found in config, using default value of 0.00"
        self.entropy_coef = get_config_value(config, "entropy_coef", 0.00, warn_str)
        
        warn_str = "collision_buffer_size not found in config, using default value of None"
        self.collision_buffer_size = get_config_value(config, 
                                                       "collision_buffer_size",
                                                       None,
                                                       warn_str)

        warn_str = "supervision_coef not found in config, using default value of 1.0"
        self.supervision_coef = get_config_value(config, "supervision_coef", 1.0, warn_str)

        warn_str = "optim_kwargs not found in config, using default value of {}"
        self.optim_kwargs = get_config_value(config, "optim_kwargs", {}, warn_str)

        warn_str = "safety_obs_key not found in config, using default value of 'observation'"
        self.safety_obs_key = get_config_value(config, "safety_obs_key", "observation", warn_str)
        
        warn_str = "scheduler_config not found in config, using default value of None"
        self.scheduler_config = get_config_value(config, "scheduler_config", None, warn_str) 

        self.loss_value_log_keys = ["loss_safety_objective",
                                    "loss_CDF", 
                                    "loss_safety_entropy",
                                    "loss_CDF_supervised",]
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
            entropy_bonus=bool(self.entropy_coef),
            entropy_coef=self.entropy_coef,
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

        if self.collision_buffer_size is None or self.collision_buffer_size == 0:
            self.collision_buffer = []
        else:
            self.collision_buffer = ReplayBuffer(
                storage=LazyTensorStorage(max_size = self.collision_buffer_size,
                                        device=self.device),
                sampler=RandomSampler(),
            )
        if self.scheduler_config is not None:
            scheduler_name = self.scheduler_config["name"]
            if scheduler_name == "linear":
                self.scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optim,
                    start_factor=self.scheduler_config["start_factor"],
                    end_factor=self.scheduler_config["end_factor"],
                    total_iters=self.scheduler_config["total_iters"],
                )
            elif scheduler_name == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optim,
                    T_max=self.scheduler_config["T_max"],
                    eta_min=self.scheduler_config["eta_min"],
                )
            elif scheduler_name == "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optim,
                    step_size=self.scheduler_config["step_size"],
                    gamma=self.scheduler_config["gamma"],
                )
            else:
                raise ValueError(f"Scheduler {scheduler_name} not found in config.\
                                Please use one of the following: linear, cosine, step")
        else:
            self.scheduler = None
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
            if self.scheduler is not None:
                self.scheduler.step()
            if wandb.run is not None:
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
        collector.shutdown()
    def _update_collision_buffer(self, td: TensorDict):
        """
        Update the collision buffer with the current tensordict data.
        Note: We assume that the data from the collector is split into trajectories
        by split_trajs = True.
        
        Args:
            td (TensorDict): The data tensor dictionary.
        """
        if not self.collision_buffer:
            return
        states = td[self.safety_obs_key]
        collision_indcs = torch.where(
            td["next", self.primary_reward_key] <0.0
        )[:-1]
        collision_states = states[collision_indcs]
        if collision_states.shape[0] == 0:
            # No collision states to add
            return
        value_target_collision_states = (
            td["next",self.primary_reward_key][td["next",self.primary_reward_key] < 0.0]
        ).unsqueeze(-1)
        new_states = TensorDict({
            "collision_states": collision_states,
            "collision_value": value_target_collision_states,
        }, batch_size=collision_states.shape[:-1], device=self.device)
        self.collision_buffer.extend(new_states)
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
            loss_module (LossModule): The loss module.
            advantage_module (TensorDictModuleBase): The advantage module.
            optim (torch.optim.Optimizer): The optimizer.
            replay_buffer (TensorDictReplayBuffer): The replay buffer.
            eval_func (Optional[Callable[None, Dict[str, float]]): An optional evaluation function.
            
        Returns:
            Dict[str, float]: The logs.
        """
        self._update_collision_buffer(tensordict_data)
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
                if len(self.collision_buffer) > 0:
                    # The buffer should only be empty in the very beginning
                    collision_data = self.collision_buffer.sample(
                        self.sub_batch_size
                    ).to(self.device)
                    subdata.update(collision_data)
                loss_vals = self._set_gradients(loss_module, subdata)
                optim.step()
                optim.zero_grad()
                

                for key in self.loss_value_log_keys:
                    logs[key] += loss_vals[key].item()
        for key in self.loss_value_log_keys:
            logs[key] /= self.num_epochs
        for key in self.reward_keys:
            logs[key] = tensordict_data["next",key].to(torch.float32).mean().item()
        logs["step_count(average)"] = (
            tensordict_data["step_count"].max(dim=1).values.to(torch.float32).mean().item()
        )
        logs["lr"] = optim.param_groups[0]["lr"]
        if eval_func is not None:
            logs.update(eval_func())
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
        loss_vals = loss_module(tensordict)
        # rename loss value keys
        loss_vals["loss_CDF"] = loss_vals["loss_critic"]
        loss_vals["loss_safety_objective"] = loss_vals["loss_objective"]
        loss_vals["loss_safety_entropy"] = loss_vals["loss_entropy"]
        del loss_vals["loss_critic"]
        del loss_vals["loss_objective"]
        del loss_vals["loss_entropy"]
        # For simplicity, we calculate the collision loss(unsafe states) here, instead of in the
        # loss module. 
        if "collision_states" in tensordict:
            # Handle the case where the collision buffer is empty
            # (Only in the beginning of training)
            collision_states = tensordict["collision_states"]
            CDF_collision_pred = loss_module.critic_network.module(collision_states)
            loss_vals["loss_CDF_supervised"] = (
                torch.nn.MSELoss(reduction='mean')(
                    CDF_collision_pred,
                    tensordict["collision_value"],
                )
            )*self.supervision_coef
        else:
            loss_vals["loss_CDF_supervised"] = torch.tensor(0.0).to(self.device)
        loss_value = (
            loss_vals["loss_safety_objective"]
            + loss_vals["loss_CDF"]
            + loss_vals["loss_safety_entropy"]
            + loss_vals["loss_CDF_supervised"]
        )
        
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(
            loss_module.parameters(),
            self.max_grad_norm,
        )
        return loss_vals