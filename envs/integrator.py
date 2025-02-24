import torch
import numpy as np
from typing import Union, Optional, List
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import (
    BoundedTensorSpec, 
    CompositeSpec,
    UnboundedContinuousTensorSpec, 
    DiscreteTensorSpec
)
from torchrl.envs.common import EnvBase
from torchrl.envs.utils import make_composite_from_td

class SafeDoubleIntegratorEnv(EnvBase):
    """Stateless environment for discrete double integrator.
        
        The state is [x, x_dot] and the action is [u].
        The continous dynamics are:
        x1_dot = x2
        x2_dot = u
        The discrete time dynamics can be discretized exactly by assuing zero-order hold:
        x1_new = x1 + x2*dt + 0.5*u*dt^2
        x2_new = x2 + u*dt

        The safety constraint is that the state should be within the box 
        [-max_x1, max_x1]x[-max_x2, max_x2]
        The input constraints are u in [-max_input, max_input]
    """
    rng = None
    batch_locked = False
    def __init__(self, td_params=None, seed=None, device=None):
        self.device = device
        if td_params is None:
            td_params = self.gen_params(device=self.device)
        super().__init__(device=self.device)
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64, device=self.device).random_(
                generator=self.rng).item()
        self.set_seed(seed)
    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng
    @staticmethod
    def gen_params(batch_size:Optional[List[int]]=None, device=None)->TensorDictBase:
        """Returns a TensorDict with the parameters of the environment

        Args:
            batch_size (int, optional): batch size. Defaults to None.
            device (string, optional): device to use. Defaults to None.

        Returns:
            TensorDictBase: TensorDict with keys 'dt'
        """
        if batch_size is None:
            batch_size = []
        td = TensorDict({
            "params": TensorDict({
                "dt": 0.01,
                "max_input": 1.0,
                "max_x1": 1.0,
                "max_x2": 1.0,
                },[],
            )
        },
        [],device=device,
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td
    def _make_spec(self, td_params:TensorDictBase):
        """Creates the tensor specs for the environment

        Args:
            td_params (TensorDictBase): TensorDict with keys 'params'
        """
        self.observation_spec = CompositeSpec(
            x1 = UnboundedContinuousTensorSpec(
                shape=(),
                dtype=torch.float32),
            x2 = UnboundedContinuousTensorSpec(
                shape=(),
                dtype=torch.float32),
            params = self.make_composite_from_td(td_params["params"]),
            shape=(),
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            low=-td_params["params", "max_input"],
            high=td_params["params", "max_input"],
            shape=(),
            dtype=torch.float32
        )
#        self.done_spec = CompositeSpec(
#            done = DiscreteTensorSpec(
#                shape  = torch.bool,
#                n= (*td_params.shape,1),
#                dtype=2),
#            shape = ()
#        )

        # TODO: This should be a discrete spec allowing only -1 and 0. This does not exist in the API
        # and a custom spec should be created
        self.reward_spec = BoundedTensorSpec(
            low=-1.0,
            high=0.0,
            shape=(*td_params.shape,1),
            dtype=torch.float32
        )
    def make_composite_from_td(self,td:TensorDict):
        composite = CompositeSpec(
            {
                key : make_composite_from_td(tensor)
                if isinstance(tensor, TensorDict) 
                else UnboundedContinuousTensorSpec(
                    dtype=tensor.dtype,
                    device=tensor.device,
                    shape=tensor.shape
                )
                for key, tensor in td.items()
            },
            shape=td.shape
        )
        return composite
    @classmethod
    def _step(cls, tensordict: TensorDict):
        """
        Args:
            tensordict (TensorDict): dict with keys 'x1','x2' and 'action'

        Returns:
            TensorDict: dict with keys 'x1' 'x2', 'reward','terminated', 'info'
        """
        x1 = tensordict['x1']
        x2 = tensordict['x2']
        u = tensordict['action'].squeeze()
        u = torch.clamp(u, -1, 1)
        # Note that here we use the state, not the next state for the cost
        # That means that when an action makes the next state unsafe, this will
        # only be detected in the next step, and we won't set termiated to True immeadiately
        # This is a design choice, and can be changed if needed
        costs = torch.zeros_like(x1)
        costs = torch.where(cls.constraints_satisfied(x1, x2), costs, torch.tensor(1.0))

        dt = tensordict["params", "dt"]
        # Unpack state and action

        x1_new = x1 + x2*dt + 0.5*u*dt**2
        x2_new = x2 + u*dt

        # Done
        terminated = costs > 0
        done = terminated.clone()
        # Reward
        reward = -costs.view(*tensordict.shape,1)

        out = TensorDict({
            "x1": x1_new,
            "x2": x2_new,
            "params": tensordict["params"],
            "reward": reward,
            "done": done,
            "terminated": terminated,
        },
        tensordict.shape)
        return out
    def _reset(self,tensordict):
        batch_size = (
            tensordict.batch_size if tensordict is not None else self.batch_size
        )
        if tensordict is None or tensordict.is_empty():
            tensordict = self.gen_params(batch_size = batch_size)
        x1 = (
            torch.rand(tensordict.shape,generator=self.rng,device=self.device,dtype=torch.float32)
            *(2*tensordict["params","max_x1"])
            - tensordict["params","max_x1"]
        )
        x2 = (
            torch.rand(tensordict.shape,generator=self.rng,device=self.device,dtype=torch.float32)
            *(2*tensordict["params","max_x2"])
            - tensordict["params","max_x2"]
        )
        terminated= torch.zeros(batch_size,dtype=torch.bool,device=self.device)
        done = terminated.clone()
        out = TensorDict(
            {
            "x1": x1,
            "x2": x2,
            "params": tensordict["params"],
            },
        batch_size=batch_size)
        return out
    @classmethod
    def constraints_satisfied(cls, x1:torch.Tensor, x2:torch.Tensor)->torch.Tensor:
        return (x1 >= -1) & (x1 <= 1) & (x2 >= -1) & (x2 <= 1)