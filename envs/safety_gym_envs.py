import gym.vector
from tensordict import TensorDict
from torchrl.envs import EnvBase
import torch
from torchrl.data.tensor_specs import (
    CompositeSpec,
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
)
import gym
import safety_gym
import numpy as np
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform

#TODO: num_envs=None is not supported yet
class SafetyGymEnv(EnvBase):
    # We can safely use this class instead of the GymEnv class since safety gym envs
    # do not reset, other than on max steps(1000), be sure to reset before this. Any reset is done by us
    batch_locked = True
    def __init__(
        self,
        env_name: str,
        num_envs: int,
    ):
        # All gym environments must run on the CPU
        super().__init__(device=torch.device("cpu"))
        assert num_envs > 0, "num_envs must be greater than 0. SafetyGymEnv does not support unbatched envs"
        if num_envs > 16:
            # Above 16 environments, we use synchronous mode
            async_envs = False
        else:
            # Below 16 environments, we use asynchronous mode
            async_envs = True
        self._env = gym.vector.make(
            env_name,
            num_envs=num_envs,
            asynchronous=async_envs,
            disable_env_checker=False,
        )
        self.batch_size = [num_envs]
        self._make_specs()
    def _step(self, tensordict: TensorDict) -> TensorDict:
        # Extract the action from the tensordict and convert it to a numpy array
        action = tensordict.get("action")
        action = action.to(torch.float32).cpu().numpy()
        next_obs, reward, done, info = self._env.step(action)
        assert (done == False).all(), "SafetyGymEnv assumes that the environment is not reset"
        # The 'cost' field in the info dict is the 'aggregate cost', i.e the sum
        # of all costs for each object in the environment.
        # We don't differentiate between the objects and all are treated equally.
        # Therfore we only check if any of the costs are positive.
        neg_cost = torch.from_numpy(info['cost']>0).to(self.device)
        neg_cost = torch.where(neg_cost == True, torch.tensor(-1.0), torch.tensor(0.0))
        # Note that device is always CPU for gym environments
        out = TensorDict(
            {
                "observation": torch.from_numpy(next_obs).to(self.device),
                "reward": torch.from_numpy(reward).to(self.device),
                "done": torch.from_numpy(done).to(self.device),
                "terminated": torch.zeros_like(torch.from_numpy(done)).to(self.device),
                "truncated": torch.zeros_like(torch.from_numpy(done)).to(self.device),
                "neg_cost": torch.from_numpy(neg_cost).to(self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        # The method assumes that the environment is reset if a constraint is violated
        out["done"] = (out["neg_cost"] < 0)
        out["terminated"] = (out["neg_cost"] < 0)
        return out
    def _reset(self, tensordict: TensorDict) -> TensorDict:
        obs = self._env.reset()
        done = np.zeros((self.batch_size[0],1), dtype=bool)
        out = TensorDict(
            {
                "observation": torch.from_numpy(obs).to(self.device),
                "done": torch.from_numpy(done).to(self.device),
                "terminated": torch.from_numpy(done).to(self.device),
                "truncated": torch.from_numpy(done).to(self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return out
    def _set_seed(self, seed: int) -> None:
        self._env.set_seed(seed)

    def _reward_space(self, env):
        if hasattr(env, "reward_space") and env.reward_space is not None:
            return env.reward_space
    def _make_done_spec(self):  # noqa: F811
        return CompositeSpec(
            {
                "done": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
                "terminated": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
                "truncated": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
            },
            shape=self.batch_size,
        )
    def _make_specs(self, batch_size=None) -> None:
        cur_batch_size = self.batch_size if batch_size is None else torch.Size([])
        action_spec = _gym_to_torchrl_spec_transform(
            self._env.action_space,
            device=self.device,
        )
        observation_spec = _gym_to_torchrl_spec_transform(
            self._env.observation_space,
            device=self.device,
        )
        if not isinstance(observation_spec, CompositeSpec):
            observation_spec = CompositeSpec(
                observation=observation_spec, shape=cur_batch_size
            )
        elif observation_spec.shape[: len(cur_batch_size)] != cur_batch_size:
            observation_spec.shape = cur_batch_size


        reward_space = self._reward_space(self._env)
        if reward_space is not None:
            base_reward_spec = _gym_to_torchrl_spec_transform(
                reward_space,
                device=self.device,
            )
        else:
            base_reward_spec = UnboundedContinuousTensorSpec(
                shape=[1],
                device=self.device,
            )
        if batch_size is not None:
            action_spec = action_spec.expand(*batch_size, *action_spec.shape)
            base_reward_spec = base_reward_spec.expand(*batch_size, *base_reward_spec.shape)
            observation_spec = observation_spec.expand(
                *batch_size, *observation_spec.shape
            )

        self.done_spec = self._make_done_spec()
        self.action_spec = action_spec
        if base_reward_spec.shape[: len(cur_batch_size)] != cur_batch_size:
            base_reward_spec = base_reward_spec.expand(*cur_batch_size, *base_reward_spec.shape)
        else:
            base_reward_spec = base_reward_spec
        self.observation_spec = observation_spec
        self.reward_spec = CompositeSpec(
            {
                "neg_cost": BoundedTensorSpec(
                    low = -1.0,
                    high = 0.0,
                    shape=base_reward_spec.shape, dtype=torch.float32),
                "reward": base_reward_spec,
            },shape=self.batch_size
        )