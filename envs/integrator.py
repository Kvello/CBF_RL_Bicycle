import torch
import numpy as np
from typing import Union, NoneType
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
    def __init__(self, td_params=None, seed=None, device=None,max_steps:int=100):
        self.device = device
        if td_params in None:
            td_params = self.gen_params(device=self.device)
        super().__init__(device=self.device)
        self._make_spec(td_params)
        if seed is not None:
            seed = torch.empty((), dtype=torch.int64, device=self.device).random_(
                generator=self.rng).item()
        self.set_seed(seed)
    def _set_seed(self, seed:int):
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        self.rng = rng
    @staticmethod
    def gen_params(batch_size:Union[int, NoneType]=None, device=None)->TensorDictBase:
        """Returns a TensorDict with the parameters of the environment

        Args:
            batch_size (Union[int, NoneType], optional): batch size. Defaults to None.
            device (string, optional): device to use. Defaults to None.

        Returns:
            TensorDictBase: TensorDict with keys 'dt'
        """
        if batch_size is None:
            batch_size = []
        td = TensorDict({
            "params": TensorDict({
                "dt": 0.01,
                "max_steps": 100,
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
        self.observation_spec = Composite(
            x1 = BoundedTensorSpec(
                low=-td_params["params", "max_x1"], 
                high=td_params["params", "max_x1"],
                shape=(),
                dtype=torch.float64),
            x2 = BoundedTensorSpec(
                low=-td_params["params", "max_x2"], 
                high=td_params["params", "max_x2"],
                shape=(),
                dtype=torch.float64),
            params = make_composite_from_td(
                td_params["params"],
                unsqueeze_null_shapes=False
            ),
            shape=(),
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            low=-td_params["params", "max_input"],
            high=td_params["params", "max_input"],
            shape=(),
            dtype=torch.float64
        )
        # TODO: This should be a discrete spec allowing only -1 and 0. This does not exist in the API
        # and a custom spec should be created
        self.reward_spec = BoundedTensorSpec(
            low=-1.0,
            high=0.0,
            shape=(),
            dtype=torch.float64
        )
    def make_composite_from_td(td:TensorDict):
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
            }
        )
    @classmethod
    def _step(cls, tensordict: TensorDict):
        """
        Args:
            tensordict (TensorDict): dict with keys 'x1','x2' and 'action'

        Returns:
            TensorDict: dict with keys 'x1' 'x2', 'reward', 'done', 'info'
        """
        x1 = tensordict['x1']
        x2 = tensordict['x2']
        u = tensordict['action'].squeeze()
        u = torch.clamp(u, -1, 1)
        costs = torch.zeros_like(x1)
        costs = torch.where(not cls.constraints_satisfied(x1, x2), torch.tensor(1.0), costs)

        dt = tensordict["params", "dt"]
        # Unpack state and action

        x1_new = x1 + x2*dt + 0.5*u*dt**2
        x2_new = x2 + u*dt

        # Reward
        reward = -costs.view(*tensordict.shape,1)

        # Done
        done = torch.zeros_like(x1)

        out = TensorDict({
            "x1": x1_new,
            "x2": x2_new,
            "params": tensordict["params"],
            "reward": reward,
            "done": done
        })
        return out
    def _reset(self,tensordict):
        batch_size = (
            tensordict.batch_size if tensordict is not None else self.batch_size
        )
        if tensordict is None or tensordict.is_empty():
            tensordict = self.gen_params(batch_size = batch_size)
        x1 = (
            torch.rand(tensordict.shape,generator=self.rng,device=self.device)
            *(2*tensordict["params","max_x1"])
            - tensordict["params","max_x1"]
        )
        x2 = (
            torch.rand(tensordict.shape,generator=self.rng,device=self.device)
            *(2*tensordict["params","max_x2"])
            - tensordict["params","max_x2"]
        )
        out = TensorDict({
            "x1": x1,
            "x2": x2,
            "params": tensordict["params"]
        })
    @classmethod
    def constraints_satisfied(cls, x1:torch.Tensor, x2:torch.Tensor):
        return (x1 >= -1) & (x1 <= 1) & (x2 >= -1) & (x2 <= 1)