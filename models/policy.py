import torch
import torch.nn as nn
from typing import List, Type
from tensordict.nn.distributions import NormalParamExtractor

class ProbabilisticPolicyNet(nn.Module):
    def __init__(self,
                 input_size:int,
                 action_size:int,
                 device:torch.device=torch.device("cpu"),
                 layers:List[int] = [64,64],
                 activation:nn.Module = nn.ReLU()):
        super().__init__()
        self.layers = nn.Sequential()
        dims = [input_size] + layers
        for i in range(len(dims)-1):
            self.layers.add_module(f"layer_{i}",nn.Linear(dims[i],dims[i+1],device=device))
            self.layers.add_module(f"activation_{i}",activation)
        self.layers.add_module("linear",nn.Linear(dims[-1],2*action_size,device=device))
        self.layers.add_module("output",NormalParamExtractor())
    def forward(self, x:torch.Tensor):
        return self.layers(x)