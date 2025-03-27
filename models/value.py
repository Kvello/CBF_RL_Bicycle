import torch
import torch.nn as nn
from typing import List, Type

class SafetyValueFunctionBase(nn.Module):
    """Base class for different Safety Value Function architectures."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("Forward method must be implemented in subclass")

class FfSafetyValueFunction(SafetyValueFunctionBase):
    """
    A simple feedforward neural network that represents the value function that is
    supposed to encode safety. The safety-preserving task structure should be used with this 
    value function. 
    This means the value should lie between -1 and 0 for all states.
    Several NN-parameterizations are possible, but the default is a 2-layer feedforward network
    with tanh activations.
    It is reasonable to suspect that the choice of parameterization will affect the
    learned CBF/value function significantly.
    """
    def __init__(self,
                 input_size:int,
                 device:torch.device=torch.device("cpu"),
                 layers:List[int] = [64,64],
                 activation:nn.Module = nn.ReLU(),
                 bounded:bool = True,
                 eps = 1e-2):
        super().__init__()
        self.layers = nn.Sequential() 
        self.bounded = bounded
        self.eps = eps
        self.device = device
        dims = [input_size] +layers 
        for i in range(len(dims)-1):
            self.layers.add_module(f"layer_{i}",nn.Linear(dims[i],dims[i+1],device=device))
            self.layers.add_module(f"activation_{i}",activation)
        self.layers.add_module("output",nn.Linear(dims[-1],1,device=device))
    def forward(self, x:torch.Tensor):
        ff = self.layers(x)
        if self.bounded:
            return -torch.sigmoid(ff)*(1+self.eps) + self.eps
        else:
            return ff

class QuadraticSafetyValueFunction(SafetyValueFunctionBase):
    """
    A simple quadratic value function that represents the value function that as
    x^T P(x) x, where P is a positive semi-definite matrix P(x)= N^T N, where N is 
    the network output.
    """
    def __init__(self,
                 input_size:int,
                 device:torch.device=torch.device("cpu"),
                 layers:List[int] = [64,64],
                 activation:nn.Module = nn.ReLU(),
                 eps = 1e-2):
        assert layers[-1] % input_size == 0, "Last layer must be a multiple of input size"
        super().__init__()
        self.eps = eps
        self.input_size = input_size
        self.layer_sizes = layers
        self.layers = nn.Sequential()
        self.device = device
        dims = [input_size] + layers
        for i in range(len(dims)-1):
            self.layers.add_module(f"layer_{i}",nn.Linear(dims[i],dims[i+1],device=device))
            self.layers.add_module(f"activation_{i}",activation)
            
    def forward(self, x:torch.Tensor):
        x_vec = x.unsqueeze(-1)
        N_x = self.layers(x).reshape(*x.shape[:-1],
                                     self.layer_sizes[-1]//self.input_size,
                                     self.input_size)
        P_x = torch.matmul(N_x.transpose(-2,-1),N_x)
        xPx = -torch.matmul(x_vec.transpose(-2,-1),torch.matmul(P_x,x_vec)).squeeze(-1)
        return xPx*(1+self.eps) + self.eps

