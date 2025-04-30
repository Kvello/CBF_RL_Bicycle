import torch
import torch.nn as nn
from typing import List
from .ff import Ff
from tensordict.nn.distributions import NormalParamExtractor

class PolicyBase(nn.Module):
    """Base class for different Value Function architectures."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("Forward method must be implemented in subclass")

class FfPolicy(PolicyBase):
    def __init__(self,
                 input_size:int,
                 device:torch.device=torch.device("cpu"),
                 layers:List[int] = [64,64],
                 activation:nn.Module = nn.ReLU()):
        Ff = Ff(self,
                    input_size=input_size,
                    device=device,
                    layers=layers,
                    activation=activation)
        self.layer_sizes = layers
        self.net = nn.Sequential()
        self.net.add_module("Ff",Ff)
        self.net.add_module("param_extractor", NormalParamExtractor())
    def forward(self, x:torch.Tensor):
        return self.net(x)