import torch
from torch import nn
import numpy as np
from typing import Union, Optional, List
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import (
    BoundedTensorSpec, 
    CompositeSpec,
    UnboundedContinuousTensorSpec, 
    DiscreteTensorSpec
)
from torchrl.envs import (
    EnvBase,
    Compose,
    TransformedEnv,
    
)
from torchrl.envs.common import EnvBase
from torchrl.envs.utils import make_composite_from_td
from torchrl.envs.transforms import Transform
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from math import ceil
from datetime import datetime

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
            td_params = TensorDict({
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
        self._params = td_params
        super().__init__(device=self.device)
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64, device=self.device).random_(
                generator=self.rng).item()
        self.set_seed(seed)
    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng
    def gen_params(self,batch_size:Optional[List[int]]=None, device=None)->TensorDictBase:
        """Returns a TensorDict with the parameters of the environment

        Args:
            batch_size (int, optional): batch size. Defaults to None.
            device (string, optional): device to use. Defaults to None.

        Returns:
            TensorDictBase: TensorDict with keys 'dt'
        """
        if batch_size is None:
            batch_size = []
        td = self._params.clone()
        td.to(device)
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
            shape=(1,),
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

    @property
    def obs_size_unbatched(self):
        return 2

        
# TODO:
# Implement the plotting functionality in a cleaner way
# Especially the plotting of trajectories ontop of the value function landscape
# Passing the value function as an argument is not ideal. Maybe passing the image 
# of the value function landscape would be better
# or maybe some other design choice would be better.
# The key is that the plotting funcitons should be separate from each other, but can
# be called in sequence to generate the desired plots

def plot_integrator_trajectories(env, 
                                 policy_module, 
                                 rollout_len:int, 
                                 num_trajectories:int,
                                 max_x1:float,
                                 max_x2:float,
                                 value_landscape:Optional[np.ndarray]=None):
    """Plots the trajectories of the agent in the environment.

    Args:
        env (_type_): _description_
        policy_module (_type_): _description_
        rollout_len (int): _description_
        num_trajectories (int): _description_
    """

    fig = plt.figure()
    i=0
    print(f"Generating {num_trajectories} trajectories")
    while i < num_trajectories:
        rollouts =  env.rollout(rollout_len, 
                                policy_module, 
                                auto_reset=True,
                                break_when_any_done=False)
        j = rollouts.shape[0]
        num_left = num_trajectories - i
        i += j
        for k in range(min(j,num_left)):
            traj_length = torch.where(rollouts["next","done"][k])[0][0] + 1
            traj_x = rollouts["x1"][k].cpu().detach().numpy()[0:traj_length]
            traj_y = rollouts["x2"][k].cpu().detach().numpy()[0:traj_length]
            plt.plot(traj_x, traj_y)
            plt.plot(traj_x[0], traj_y[0], 'og')
            if traj_length < rollout_len-1:
                plt.plot(traj_x[-1], traj_y[-1], 'xr')
    plt.plot([-max_x1, -max_x1], [-max_x2, max_x2], "r")
    plt.plot([max_x1, max_x1], [-max_x2, max_x2], "r")
    plt.plot([-max_x1, max_x1], [-max_x2, -max_x2], "r")
    plt.plot([-max_x1, max_x1], [max_x2, max_x2], "r")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Trajectories of the agent")
    plt.xlim(-max_x1*1.1, max_x1*1.1)
    plt.ylim(-max_x2*1.1, max_x2*1.1)
    if value_landscape is not None:
        levels = [0.0]
        x_vals = np.linspace(-max_x1*1.1, max_x1*1.1, value_landscape.shape[0])
        y_vals = np.linspace(-max_x2*1.1, max_x2*1.1, value_landscape.shape[1])
        mesh = np.meshgrid(x_vals, y_vals)
        locator = MaxNLocator(nbins=10)
        levels_locator = locator.tick_values(value_landscape.min(), value_landscape.max())
        plt.contour(mesh[0],mesh[1],value_landscape,levels=levels,colors="black")
        plt.contourf(mesh[0],mesh[1],value_landscape,levels=levels_locator,cmap='coolwarm')
        plt.colorbar()
    plt.savefig("results/integrator_trajectories" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pdf")
def plot_value_function_integrator(max_x1:float, max_x2:float,
                                resolution:int, 
                                value_net:nn.Module,
                                levels:List[float] = [0.0])->np.ndarray:
    """Plots the value function landscape across the state space.
    Current implementation only supports 2D state spaces.
    Args:
        state_space dict: A dictionary containing a representation of the state space over
            which the value function should be evaluated. The dictionary should contain
            the keys "low" and "high" for each state "state_name"
            which represent the lower and upper bounds of the
            state space respectively. E.g state_space = {"state_name": {"low": -1, "high": 1}}
        value_module nn.Module: The value function module.
        resolution (int): The resolution of the grid over which 
        the value function should be evaluated. Number of points per unit.
    Returns:
        np.ndarray: The value function values evaluated over the state space.
    """
    x1_low = -max_x1*1.1
    x1_high = max_x1*1.1
    x2_low = -max_x2*1.1
    x2_high = max_x2*1.1
    linspace_x1 = torch.linspace(x1_low, x1_high, ceil(resolution*(x1_high-x1_low)))
    linspace_x2 = torch.linspace(x2_low, x2_high, ceil(resolution*(x2_high-x2_low)))
    linspaces = [linspace_x1, linspace_x2]
    mesh = torch.meshgrid(*linspaces,indexing="xy")
    inputs = torch.stack([m.flatten() for m in mesh],dim=-1)
    outputs = value_net(inputs).detach().cpu().numpy()
    outputs = outputs.reshape(mesh[0].shape)
    fig = plt.figure()

    locator = MaxNLocator(nbins=10)
    levels_locator = locator.tick_values(outputs.min(), outputs.max())
    plt.contour(mesh[0],mesh[1],outputs,levels=levels,colors="black")
    plt.contourf(mesh[0],mesh[1],outputs,levels=levels_locator,cmap='coolwarm')
    plt.plot([-max_x1, -max_x1], [-max_x2, max_x2], "r")
    plt.plot([max_x1, max_x1], [-max_x2, max_x2], "r")
    plt.plot([-max_x1, max_x1], [-max_x2, -max_x2], "r")
    plt.plot([-max_x1, max_x1], [max_x2, max_x2], "r")
    plt.xlim(-max_x1*1.1, max_x1*1.1)
    plt.ylim(-max_x2*1.1, max_x2*1.1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Value function landscape")
    plt.colorbar()
    plt.savefig("results/value_function_landscape" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pdf")
    return outputs