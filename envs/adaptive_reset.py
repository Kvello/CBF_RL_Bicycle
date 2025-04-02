from torchrl.envs import EnvBase
from torchr.date import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from typing import Callable
from tensordict import TensorDict
import torch

class AdaptiveResetWrapper(EnvBase):
    """Environment wrapper that biases initial states 
    using a buffer."""
    def __init__(self, 
                 buffer_size:int,
                 env,
                 save_fn:Callable[TensorDict,bool], 
                 buffer_fraction=0.5):
        super().__init__()
        self.env = env
        self.buffer = ReplayBuffer(
            storage=LazyTensorStorage(size=buffer_size),
            sampler=SamplerWithoutReplacement()
        )
        self.buffer_fraction = buffer_fraction
        self.save_fn = save_fn
    def _reset(self,tensordict:TensorDict)->TensorDict:
        batch_size = (
            tensordict.batch_size if tensordict is not None else self.batch_size
        )
        if tensordict is None or tensordict.is_empty():
            tensordict = self.gen_params(batch_size = batch_size)
        # Save to buffer any initial states that fulfill the condition
        save_indcs = self.save_fn(tensordict)
        self.buffer.extend(tensordict[save])
        # Sample from buffer
        num_buffer_samples = min(len(self.buffer), int(self.buffer_fraction*batch_size))
        return out
    def __getattr__(self, name):
        """Automatically forward method calls to the wrapped environment."""
        return getattr(self.env, name)
    