from tensordict import TensorDict
from .custom_GymEnv import custom_GymEnv
from torchrl.envs import EnvBase
import torch
from torchrl.data.tensor_specs import (
    CompositeSpec,
    BoundedTensorSpec,
)
from typing import Optional

class SafetyGymEnv(EnvBase):
    batch_locked = True
    def __init__(
        self,
        env_name: str,
        device:Optional[torch.device]=None,
        num_envs: Optional[int] = None,
    ):
        super().__init__(device=device)
        self._env = custom_GymEnv(env_name, device=device, num_envs=num_envs)
        self.device = device
        if num_envs is not None:
            self.batch_size = [num_envs]
        else:
            self.batch_size = []
        self._make_specs()
    def _step(self, tensordict: TensorDict) -> TensorDict:
        # Call the base class step
        out = self._env._step(tensordict)
        # Convert the cost to a tensor and add it to the tensordict
        # The 'cost' field in the info dict is the 'aggregate cost', i.e the sum
        # of all costs for each object in the environment.
        # We don't differentiate between the objects and all are treated equally.
        # Therfore we only check if any of the costs are positive.
        new_vals = TensorDict({
            "neg_cost": -(out.get('cost')>0).view_as(out.get("reward")).to(torch.float32)
        },out.batch_size, device=self.device)
        out.update(new_vals)
        # Discard the other fields from the info dict
        # and only keep the 'neg_cost' field
        out_keys = ["neg_cost","observation","reward","done","terminated","truncated"]
        out = TensorDict({
            key:out[key] for key in out_keys
        },out.batch_size, device=self.device)
        return out
    def _reset(self, tensordict: TensorDict) -> TensorDict:
        out = self._env._reset(tensordict)
        return out
    def _set_seed(self, seed: int) -> None:
        self._env.set_seed(seed)
    def _make_specs(self, batch_size=None) -> None:
        # Add the 'neg_cost' field to the spec
        self._env._make_specs(self._env._env)
        base_reward_spec = self._env.reward_spec    
        self.observation_spec = self._env.observation_spec
        self.action_spec = self._env.action_spec
        self.done_spec = self._env.done_spec
        self.state_spec = self._env.state_spec
        self.reward_spec = CompositeSpec(
            {
                "neg_cost": BoundedTensorSpec(
                    low = -1.0,
                    high = 0.0,
                    shape=base_reward_spec.shape, dtype=torch.float32),
                "reward": base_reward_spec,
            },shape=self.batch_size
        )