import torch
import numpy as np
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tesor_specs import Bounded, Composite, Unbounded
from torchrl.envs.common import EnvBase
from torchrl.envs.utils import make_composite_from_td

class SafeBicycleEnv(EnvBase):
    """
    Stateless environment for the velocity controlled bicycle model.
    """
    constraint_radius = 2.0
    @classmethod
    def _step(cls, tensordict: TensorDict):
        """
        Args:
            tensordict (TensorDict): dict with keys 'state' and 'action'

        Returns:
            TensorDict: dict with keys 'state', 'reward', 'done', 'info'
        """
        state = tensordict['state']
        action = tensordict['action']
        dt = tensordict["params", "dt"]
        # Unpack state and action
        x, y, theta, phi = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        v, omega = action[:, 0], action[:, 1]

        v = v.clamp(-tensordict["params", "v_max"], tensordict["params", "v_max"])
        omega = omega.clamp(-tensordict["params", "omega_max"], tensordict["params", "omega_max"])
        cost = 1.0 if not cls.state_constraints_satisfied(state) else 0.0 

        # Update state
        x_new += v * torch.cos(theta)*cls.dt
        y_new += v * torch.sin(theta)*cls.dt
        theta_new += omega*cls.dt
        phi_new = omega*cls.dt

        # Reward
        reward = -cost.view(*tensordict.shape,1)

        # Done
        done = cost > 0.0

        out = TensorDict({
            "state": torch.stack([x_new, y_new, theta_new, phi_new], dim=-1),
            "params": tensordict["params"],
            "reward": reward,
            "done": done
        })
        return out
    @classmethod
    def _reset(cls, tensordict: TensorDict):
        """
        Args:
            tensordict (TensorDict): dict with keys 'state'

        Returns:
            TensorDict: dict with keys 'state'
        """
        if tensordict is None or tensordict.is_empty():
            tensordict = self.gen_params(batch_size = cls.batch_size)
        # Sample randomly from the constraint circle. 
        x = torch.rand(tensordict["params", "batch_size"]) * 2*cls.constraint_radius - cls.constraint_radius
        y = torch.rand(tensordict["params", "batch_size"]) * 2*cls.constraint_radius - cls.constraint_radius
        theta = torch.rand(tensordict["params", "batch_size"]) * 2*np.pi - np.pi
        phi = torch.rand(tensordict["params", "batch_size"]) * 2*np.pi - np.pi
        return TensorDict({
            "state": torch.stack([x, y, theta, phi], dim=-1)
        })
    @staticmethod
    def gen_params(batch_size: int = None,device=None):
        td = TensorDict({
            "params": TensorDict(
                {
                "dt": 0.1,
                "v_max": 1.0,
                "omega_max": 1.0
                },
                [],
            )
        },
        [],
        device=device
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    @classmethod
    def state_constraints_satisfied(cls, state):
        x, y, _,_ = state
        return x**2 + y**2 <= cls.constraint_radius**2