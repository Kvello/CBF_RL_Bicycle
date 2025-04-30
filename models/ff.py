import torch
import torch.nn as nn
from typing import List

    
class Ff(nn.Module):
    """
    A simple feedforward neural network
    """
    def __init__(self,
                 input_size:int,
                 device:torch.device=torch.device("cpu"),
                 layers:List[int] = [64,64],
                 activation:nn.Module = nn.ReLU()):
        super().__init__()
        self.layers = nn.Sequential() 
        self.device = device
        dims = [input_size] +layers[:-1]
        for i in range(len(dims)-1):
            self.layers.add_module(f"layer_{i}",nn.Linear(dims[i],dims[i+1],device=device))
            self.layers.add_module(f"activation_{i}",activation)
        self.layers.add_module("output",nn.Linear(dims[-1],layers[-1],device=device))
    def forward(self, x:torch.Tensor):
        ff = self.layers(x)
        return ff