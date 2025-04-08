from torchrl.envs import EnvBase
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from typing import Callable, Optional
from tensordict import TensorDict
import torch

class AdaptiveResetEnv(EnvBase):
    """Environment wrapper that biases initial states 
    using a buffer."""
    def __init__(self, 
                 buffer_size,
                 buffer_fraction=0.5,
                 **kwargs):
        super().__init__()
        self.buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=buffer_size),
            sampler=SamplerWithoutReplacement()
        )
        self.buffer_fraction = buffer_fraction
    def extend_initial_state_buffer(self,td:TensorDict):
        """Extends the initial state buffer with the provided tensor dict."""
        raise NotImplementedError("This method should be implemented in a subclass.")