import torch
from torch import nn
import numpy as np
from typing import Union, Optional, List, Callable
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
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.collectors.utils import split_trajectories
import warnings
        
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
    batch_locked = True
    def __init__(self,
                 batch_size:Union[int,None]=None, 
                 td_params=None, 
                 seed=None, 
                 device=None,
                 buffer_reset_fraction:float=0.0):
        self.device = device
        if batch_size is None:
            self.batch_size = []
        else:
            self.batch_size = [batch_size]
        if td_params is None:
            # Default parameters
            td_params = TensorDict({
                "dt": 0.01,
                "max_input": 1.0,
                "max_x1": 1.0,
                "max_x2": 1.0,
                    },[],device=device)
        self._params = td_params
        super().__init__(device=self.device)
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64, device=self.device).random_(
                generator=self.rng).item()
        self.set_seed(seed)
        self.buffer_reset_fraction = buffer_reset_fraction
        if self.buffer_reset_fraction > 0.0 :
            self._initial_state_buffer = ReplayBuffer(
                storage=LazyTensorStorage(max_size=self.batch_size[0]*buffer_reset_fraction),
                sampler=SamplerWithoutReplacement()
            )
        self._eval = False
    def _set_seed(self, seed: Optional[int]):
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        self.rng = rng
    @property
    def params(self):
        return self._params
    @property
    def evaluate(self):
        return self._eval
    @evaluate.setter
    def evaluate(self, value:bool):
        self._eval = value
    def _make_spec(self, td_params:TensorDictBase):
        """Creates the tensor specs for the environment

        Args:
            td_params (TensorDictBase): TensorDict with keys 'params'
        """
        self.observation_spec = CompositeSpec(
            x1 = UnboundedContinuousTensorSpec(
                shape=self.batch_size,
                dtype=torch.float32),
            x2 = UnboundedContinuousTensorSpec(
                shape=self.batch_size,
                dtype=torch.float32),
            shape=self.batch_size,
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            low=-td_params["max_input"],
            high=td_params["max_input"],
            shape=(*self.batch_size,1),
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
            shape=(*self.batch_size,1),
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
    def _step(self, tensordict: TensorDict):
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
        params = self._params
        rewards = torch.ones_like(x1)
        rewards = torch.where(
            SafeDoubleIntegratorEnv.constraints_satisfied(params,x1, x2), 
            rewards, 
            torch.tensor(0.0,device=x1.device)
        )

        dt = params["dt"]
        # Unpack state and action

        x1_new = x1 + x2*dt + 0.5*u*dt**2
        x2_new = x2 + u*dt

        # Done
        terminated = rewards == 0.0
        done = terminated.clone()
        # Reward
        reward = rewards.view(*tensordict.shape,1)

        out = TensorDict({
            "x1": x1_new,
            "x2": x2_new,
            "reward": reward,
            "done": done,
            "terminated": terminated,
        },
        tensordict.shape)
        return out
    def _reset(self,tensordict):
        params = self._params

        x1 = (
            torch.rand(self.batch_size,generator=self.rng,device=self.device,dtype=torch.float32)
            *(2*params["max_x1"])
            - params["max_x1"]
        )
        x2 = (
            torch.rand(self.batch_size,generator=self.rng,device=self.device,dtype=torch.float32)
            *(2*params["max_x2"])
            - params["max_x2"]
        )
        terminated= torch.zeros(self.batch_size,dtype=torch.bool,device=self.device)
        done = terminated.clone()
        if self.buffer_reset_fraction > 0.0 and self._eval == False:
            # Sample from the buffer
            num_buffer_samples = min(
                len(self._initial_state_buffer),
                int(self.buffer_reset_fraction*self.batch_size[0])
            )
            if num_buffer_samples > 0:
                buffer_samples = self._initial_state_buffer.sample(num_buffer_samples)
                x1_buffer = buffer_samples["x1"]
                x2_buffer = buffer_samples["x2"]
                # We always place the samples from the buffer in the end of the batch
                # This way, any trajectories starting from an initial point from the buffer
                # that terminate will again be started from an initial point from the buffer.
                # This ensures we always have a proportion equal to 'buffer_fraction'
                # of the steps from samples that started from initial points in the buffer
                x1[-num_buffer_samples:] = x1_buffer
                x2[-num_buffer_samples:] = x2_buffer
        out = TensorDict(
            {
            "x1": x1,
            "x2": x2,
            },
        batch_size=self.batch_size)
        return out
    @staticmethod
    def constraints_satisfied(params:TensorDict, 
                              x1:torch.Tensor, 
                              x2:torch.Tensor)->torch.Tensor:
        """Returns a boolean tensor indicating whether the constraints are satisfied
        
        Args:
            params (TensorDict): The parameters of the environment
            x1 (torch.Tensor): The state x1
            x2 (torch.Tensor): The state x2
            
        Returns:
            torch.Tensor: A boolean tensor indicating whether the constraints are satisfied
        """
        max_x1 = params["max_x1"]
        max_x2 = params["max_x2"]
        return (x1 >= -max_x1) & (x1 <= max_x1) & (x2 >= -max_x2) & (x2 <= max_x2)

    def extend_initial_state_buffer(self,td:TensorDict):
        """Extends the initial state buffer with the initial states that resulted in
        a safety violation. This is used to bias the initial states of the environment
        to be more likely to be in the unsafe region.

        Args:
            td (TensorDict): The tensordict of rollouts from the environment. E.g.
            collected from a collector.

        """
        if self.buffer_reset_fraction == 0.0:
            # warnings.warn("Adaptive reset is not enabled. Cannot extend initial state buffer.")
            return
        if td.batch_size[0] == self.batch_size[0]:
            # Splitting has likely not been done yet
            trajs = split_trajectories(td)
        else:
            trajs = td
        collision_traj_ids = torch.where(trajs["next","reward"] < 0.0)[0]
        initial_states = TensorDict({
            "x1": trajs[collision_traj_ids,0]["x1"].reshape(-1),
            "x2": trajs[collision_traj_ids,0]["x2"].reshape(-1)}
            ,batch_size=collision_traj_ids.shape,
            device=td.device)
        #Ignore the initial states that are too similar to the ones in the buffer
        # This could be implemented more efficiently
        if len(self._initial_state_buffer) == 0:
            self._initial_state_buffer.extend(initial_states)
            return
        new_vals = torch.stack(
            [initial_states["x1"], initial_states["x2"]],
            dim=-1
        ).unsqueeze(0)
        buffer_vals = torch.stack(
            [self._initial_state_buffer["x1"], self._initial_state_buffer["x2"]],
            dim=-1
        ).unsqueeze(0)
        dists = torch.cdist(new_vals, buffer_vals).squeeze(0)
        min_dists = dists.min(dim=-1)[0]
        mask = min_dists > 0.1
        initial_states = initial_states[mask]
        if initial_states.shape[0] == 0:
            return
        self._initial_state_buffer.extend(initial_states)
        
    def plot_initial_state_buffer(self,ax):
        """Plots the initial state buffer. This is used to visualize the
        initial states that have been collected in the buffer.
        """
        if self.buffer_reset_fraction == 0.0:
            warnings.warn("Adaptive reset is not enabled. Cannot plot initial state buffer.")
            return
        x1 = self._initial_state_buffer["x1"].cpu().detach().numpy()
        x2 = self._initial_state_buffer["x2"].cpu().detach().numpy()
        print("Plotting initial state buffer")
        print(f"it has {len(x1)} samples")
        print(f"x1: {x1}, x2: {x2}")
        ax.plot(x1, x2,'ob')
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Initial state buffer")
        ax.set_xlim(-self.params["max_x1"]*1.1, self.params["max_x1"]*1.1)
        ax.set_ylim(-self.params["max_x2"]*1.1, self.params["max_x2"]*1.1)
        ax.plot([-self.params["max_x1"], -self.params["max_x1"]], [-self.params["max_x2"], self.params["max_x2"]], "r")
        ax.plot([self.params["max_x1"], self.params["max_x1"]], [-self.params["max_x2"], self.params["max_x2"]], "r")
        ax.plot([-self.params["max_x1"], self.params["max_x1"]], [-self.params["max_x2"], -self.params["max_x2"]], "r")
        ax.plot([-self.params["max_x1"], self.params["max_x1"]], [self.params["max_x2"], self.params["max_x2"]], "r")

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
                                policy_module: nn.Module,
                                rollout_len:int, 
                                num_trajectories:int,
                                value_net:Optional[nn.Module]=None,
                                levels:List[float] = [0.0]):
    """Plots the trajectories of the agent in the environment.

    Args:
        env (EnvBase): The environment.
        policy_module (nn.Module): The policy module.
        rollout_len (int): The length of the rollouts.
        num_trajectories (int): The number of trajectories to plot.
        max_x1 (float): The maximum value of x1.
        max_x2 (float): The maximum value of x2.
        value_net (nn.Module, optional): The value function module. Defaults to None.
        if provided will be used to plot the value function landscape as a countour plot
        under the trajectories.
    """

    fig = plt.figure()
    i=0
    max_x1 = env.params["max_x1"]
    max_x2 = env.params["max_x2"]
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
    env.plot_initial_state_buffer(plt.gca())
    if value_net is not None:
        x1_low = -max_x1*1.1
        x1_high = max_x1*1.1
        x2_low = -max_x2*1.1
        x2_high = max_x2*1.1
        resolution = 10
        x_vals = torch.linspace(x1_low, x1_high, ceil(resolution*(x1_high-x1_low)))
        y_vals = torch.linspace(x2_low, x2_high, ceil(resolution*(x2_high-x2_low)))
        mesh = torch.meshgrid(x_vals, y_vals)
        inputs = torch.stack([m.flatten() for m in mesh],dim=-1)
        value_function_landscape = value_net(inputs).detach().cpu().numpy()
        value_landscape = value_function_landscape.reshape(mesh[0].shape)
        
        locator = MaxNLocator(nbins=10)
        levels_locator = locator.tick_values(value_landscape.min(), value_landscape.max())
        plt.contour(mesh[0],mesh[1],value_landscape,levels=levels,colors="black")
        plt.contourf(mesh[0],mesh[1],value_landscape,levels=levels_locator,cmap='coolwarm')
        plt.colorbar()
    plt.savefig("results/integrator_trajectories" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pdf")
        
def plot_value_function_integrator(max_x1:float, max_x2:float,
                                resolution:int, 
                                value_net:nn.Module,
                                levels:List[float] = [0.0]):
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

    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(mesh[0],mesh[1],outputs,cmap='coolwarm')
    ax.plot([-max_x1, -max_x1], [-max_x2, max_x2],[0,0], "r")
    ax.plot([max_x1, max_x1], [-max_x2, max_x2],[0,0], "r")
    ax.plot([-max_x1, max_x1], [-max_x2, -max_x2],[0,0], "r")
    ax.plot([-max_x1, max_x1], [max_x2, max_x2], [0,0],"r")
    ax.set_xlim(-max_x1*1.1, max_x1*1.1)
    ax.set_ylim(-max_x2*1.1, max_x2*1.1)
   
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Value function") 
    ax.set_title("Value function landscape")
    fig.colorbar(surf)
    plt.savefig("results/value_function_landscape" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pdf")

class MultiObjectiveDoubleIntegratorEnv(SafeDoubleIntegratorEnv):
    """Stateless environment for discrete double integrator with two reward signals.
    One encouraging safety, the other any other type of goal.
        
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
    primary_reward_key:str = "r1"
    secondary_reward_key:str = "r2"
    def __init__(self,**kwargs):
        batch_locked = True
        super().__init__(**kwargs)
    @classmethod
    # For some reason creating the secondary reward function as a modifiable class attribute 
    # does not work with the multisync collector. Instead, the collecor uses the default function
    # regardless of modifications to the class attribute. Therefore, the function is
    # defined as a static method and called from the step method.
    @staticmethod
    def _secondary_reward_func(x1:torch.Tensor, x2:torch.Tensor)->torch.Tensor:
        return x1*x2
    def _step(self, tensordict: TensorDict)->TensorDict:
        r2 = torch.as_tensor(
            MultiObjectiveDoubleIntegratorEnv._secondary_reward_func(tensordict["x1"],
                                                                     tensordict["x2"])
        )
        out = super()._step(tensordict)
        r1 = out["reward"].clone()
        r2 = r2.view_as(r1).to(torch.float32)
        new_vals = TensorDict({
            MultiObjectiveDoubleIntegratorEnv.primary_reward_key: r1,
            MultiObjectiveDoubleIntegratorEnv.secondary_reward_key: r2,
        },out.batch_size)
        out.update(new_vals)
        return out
    def _make_spec(self, td_params:TensorDictBase):
        super()._make_spec(td_params)
        base_reward_spec = self.reward_spec
        self.reward_spec = CompositeSpec(
            {
                "reward": base_reward_spec,
                self.primary_reward_key: base_reward_spec,
                self.secondary_reward_key: UnboundedContinuousTensorSpec(
                    shape=(*self.batch_size,1),
                    dtype=torch.float32
                )
            },
            shape=self.batch_size,
        )