from collections import defaultdict
import torch
from typing import List, Dict, Any, Optional, Callable
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from torchrl.collectors import DataCollectorBase
from torchrl.data.replay_buffers import (
    TensorDictReplayBuffer,
    ReplayBuffer,
    LazyTensorStorage,
    RandomSampler,
)
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

def gradient_projection(
    common_module:torch.nn.Module,
    primary_loss:torch.Tensor,
    secondary_loss:torch.Tensor)->torch.Tensor:
        r"""Calculates a projected gradient of the secondary loss onto the nullspace
        of the primary loss. Returns the sum of the primary loss gradient and the
        projected secondary loss gradient. I.e
        .. math::
            \begin{align}
                \boldsymbol{\eta} &= \nabla_{\theta} L_2^{\text{CLIP}}(\theta) - \text{Proj}_{\nabla_{\theta} L_1^{\text{CLIP}}(\theta)}(\nabla_{\theta} L_2^{\text{CLIP}}(\theta))\\
                \boldsymbol{\delta_\pi} &= \nabla_{\theta} L_1^{\text{CLIP}}(\theta) + w^{\text{CLIP}}\boldsymbol{\eta}
                w^{\text{CLIP}} &= l_r \cdot \frac{||\nabla_{\theta} L_1^{\text{CLIP}}(\theta)||}{||\boldsymbol{\eta}||}
            \end{align}
        Where l_r is the output of the gradient_scaler function, and becomes the relative length
        between the primary and projected secondary loss gradients. Theta are the parameters of the
        common_module
        
        Args:
            common_module (torch.nn.Module): Common module with which the two losses were computed 
            containing the parameters
            primary_loss (torch.Tensor): The primary loss tensor
            secondary_loss (torch.Tensor): The secondary loss tensor
        Returns:
            torch.Tensor: The projected and combined gradient
            
        Note:
            Calling this funciton clears the gradients of the common module,
            and does not retain the computational graph after the projection.
            This means that to recompute the gradients, the losses have to be recomputed
            as well.
        """
        # Primary objective loss gradient
        primary_loss.backward(retain_graph=True)
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
        if torch.isclose(grad_vec_primary_loss.norm(),torch.tensor(0.0),atol=1e-10):
            # If primary loss gradient is zero, return secondary loss gradient
            # This is an edge case, and should not happen in practice
            return grad_vec_secondary_loss
        if torch.dot(grad_vec_secondary_loss,grad_vec_primary_loss) < 0.0:
            # Projection of secondary objective loss gradient onto nullspace of
            # primary objective loss gradient
            secondary_proj =(
                torch.dot(grad_vec_secondary_loss, grad_vec_primary_loss)
                / torch.dot(grad_vec_primary_loss, grad_vec_primary_loss)
                * grad_vec_primary_loss
            )
            secondary_proj = grad_vec_secondary_loss - secondary_proj 
        else:
            secondary_proj = grad_vec_secondary_loss
        grad = secondary_proj + grad_vec_primary_loss
        return grad

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
        self.device = get_config_value(config, "device", torch.device("cpu"), warn_str)

        warn_str = "clip_epsilon not found in config, using default value of 0.2"
        self.clip_epsilon = get_config_value(config, "clip_epsilon", 0.2, warn_str)
        
        warn_str = "lmbda1 not found in config, using default value of 0.95"
        self.lmbda1 = get_config_value(config, "lmbda1", 0.95, warn_str)

        warn_str = "lmbda2 not found in config, using default value of 0.95"
        self.lmbda2 = get_config_value(config, "lmbda2", 0.95, warn_str)

        warn_str = "critic_coef not found in config, using default value of 1.0"
        self.critic_coef = get_config_value(config, "critic_coef", 1.0, warn_str)

        warn_str = "gamma not found in config, using default value of 0.99"
        self.gamma = get_config_value(config, "gamma", 0.99, warn_str)

        warn_str = "num_epochs not found in config, using default value of 10"
        self.num_epochs = get_config_value(config, "num_epochs", 10, warn_str) 
        
        warn_str = "primary_reward_key not found in config, using default value of 'r1'"
        self.primary_reward_key = get_config_value(config, "primary_reward_key", "r1", warn_str)
        
        warn_str = "secondary_reward_key not found in config, using default value of 'r2'"
        self.secondary_reward_key = get_config_value(config, "secondary_reward_key", "r2", warn_str)

        warn_str = "entropy_coef not found in config, using default value of 0.0"
        self.entropy_coef = get_config_value(config, "entropy_coef", 0.0, warn_str)

        warn_str = "collision_buffer_size not found in config, using default value of 1e6"
        self.collision_buffer_size = get_config_value(config, 
                                                      "collision_buffer_size",
                                                      int(1e6),
                                                      warn_str)
        warn_str = "supervision_coef not found in config, using default value of 1.0"
        self.supervision_coef = get_config_value(config, "supervision_coef", 1.0, warn_str)

        warn_str = "optim_kwargs not found in config, using default value of {}"
        self.optim_kwargs = get_config_value(config, "optim_kwargs", {}, warn_str)

        self.loss_value_log_keys = {
            "loss_safety_objective",
            "loss_secondary_objective",
            "loss_CBF",
            "loss_CBF_supervised",
            "loss_secondary_critic",
            "loss_safety_entropy",
            "loss_secondary_entropy",
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
            
            
        self.loss_module = HiPPOLoss(
            actor=policy_module,
            primary_critic=V_primary,
            secondary_critic=V_secondary,
            clip_epsilon=self.clip_epsilon,
            critic_coef=self.critic_coef,
            supervision_coef=self.supervision_coef,
            entropy_coef=self.entropy_coef,
            gamma=self.gamma,
        )
        self.advantage_module = MultiGAE(
            gamma=self.gamma,
            lmbda1=self.lmbda1,
            lmbda2=self.lmbda2,
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
            **self.optim_kwargs
        )
        self.collision_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size = self.collision_buffer_size,
                                      device=self.device),
            sampler=RandomSampler(),
        )
        print("Training with config:")
        print(self.config)
        logs = defaultdict(list)
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
            loss_vals["loss_CBF"] +
            loss_vals["loss_secondary_critic"] +
            loss_vals["loss_CBF_supervised"]
        )
        critic_loss.backward()
        safety_loss = (
            loss_vals["loss_safety_objective"] + loss_vals["loss_safety_entropy"]
        )
        secondary_loss = (
            loss_vals["loss_secondary_objective"] + loss_vals["loss_secondary_entropy"]
        )
        policy_grad = gradient_projection(loss_module.actor_network, 
                            safety_loss, 
                            secondary_loss)
                # Set gradient to the policy module
        last_param_idx = 0
        for p in loss_module.actor_network.parameters():
            new_grad = policy_grad[last_param_idx:last_param_idx + p.data.numel()]
            new_grad = new_grad.view_as(p.data)
            p.grad = new_grad
            last_param_idx += p.data.numel()
        # this is not strictly mandatory but it's good practice to keep
        # your gradient norm bounded
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 
                                       self.max_grad_norm)
        return loss_vals
