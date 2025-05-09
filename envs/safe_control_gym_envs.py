from safe_control_gym.utils.registration import register
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from functools import partial
from torchrl.envs import EnvBase
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform
from torchrl.data.tensor_specs import (
    CompositeSpec,
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
)
import os
import torch
import numpy as np
from tensordict import TensorDict
import gymnasium as gym
from torchrl.envs import GymEnv

class OldStepAPIWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Map old API → new API
        return obs, reward, done, False, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
config_factory = ConfigFactory()
config_full_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cartpole.yaml",
)
print(f"Loading config from {config_full_path}")
config = config_factory.merge([config_full_path])
env_func = partial(make,
                    "cartpole",
                    **config.task_config)

def wrapped_env_func(**kwargs):
    env = env_func(**kwargs)
    return OldStepAPIWrapper(env)
gym.register("cartpole", wrapped_env_func)

class CartPoleEnv(EnvBase):
    '''Safe Control Gym, cart pole Environment.'''
    batch_locked = True
    def __init__(self, 
                 num_envs=1,
                 device = torch.device('cpu')):
        super().__init__(device=device)
        self.batch_size = [num_envs] if num_envs > 1 else []
        # Cannot serialize pybullet envs. Must use synchronous mode
        if num_envs > 1:
            async_envs = False
            self._env = gym.vector.make("cartpole",
                                        num_envs=num_envs,
                                        asynchronous=async_envs)
        else:
            self._env = gym.make("cartpole")
                                    
        self._make_specs()
    def _set_seed(self, seed: int) -> None:
        self._env.reset(seed=seed)
    def _reset(self, tensordict: TensorDict) -> TensorDict:
        '''Reset the environment.'''
        obs ,_ = self._env.reset()
        if self.batch_size:
            done = np.zeros((self.batch_size[0],1), dtype=bool)
        else:
            done = np.zeros((1,), dtype=bool)
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
    def _step(self, tensordict):
        '''Step the environment.'''
        action = tensordict.get("action")
        action = action.to(torch.float32).cpu().numpy()
        next_obs, reward, done,_, info = self._env.step(action)
        done = np.array(done)
        reward = np.array(reward)
        next_obs = np.array(next_obs)

        assert (done == False).all(), "CarPoleEnv assumes that the environment is not reset externally"

        constraint_violated = np.array(info['constraint_violation']==1)
        constraint_violated = torch.from_numpy(constraint_violated).to(self.device)
        neg_cost = torch.where(constraint_violated == True, torch.tensor(-1.0), torch.tensor(0.0))
        out = TensorDict(
            {
                "observation": torch.from_numpy(next_obs).to(self.device),
                "reward": torch.from_numpy(reward).to(self.device),
                "done": torch.from_numpy(done).to(self.device),
                "terminated": torch.zeros_like(torch.from_numpy(done)).to(self.device),
                "truncated": torch.zeros_like(torch.from_numpy(done)).to(self.device),
                "neg_cost": neg_cost.to(self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        # The method assumes that the environment is reset if a constraint is violated
        out["done"] = (out["neg_cost"] < 0)
        out["terminated"] = (out["neg_cost"] < 0)
        return out    
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
    def render(self):
        '''Render the environment.'''
        return self._env.render()