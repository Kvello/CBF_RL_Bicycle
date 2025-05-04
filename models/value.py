import torch
import torch.nn as nn
from typing import List
from .ff import Ff

class ValueBase(nn.Module):
    """Base class for different Value Function architectures."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("Forward method must be implemented in subclass")

class FfValueFunction(ValueBase):
    """
    A simple feedforward neural network that represents a value function that is
    Several NN-parameterization are possible, but the default is a 2-layer feedforward network
    with ReLU activations.
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
        feed_forward = Ff(input_size=input_size,
                            device=device,
                            layers=layers,
                            activation=activation)
        self.net = nn.Sequential()
        self.net.add_module("Ff",feed_forward)
        self.net.add_module("output", nn.Linear(layers[-1],1).to(device))
        self.bounded = bounded
        self.eps = eps
    def forward(self, x:torch.Tensor):
        net_out = self.net(x)
        if self.bounded:
            return -torch.sigmoid(net_out)*(1+self.eps) + self.eps
        else:
            return net_out

class QuadraticValueFunction(ValueBase):
    """
    A quadratic value function that represents the value function that as
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
        self.feed_forward = Ff(input_size=input_size,
                            device=device,
                            layers=layers,
                            activation=activation)
        self.eps = eps
        self.input_size = input_size
        self.layer_sizes = layers
            
    def forward(self, x:torch.Tensor):
        x_vec = x.unsqueeze(-1)
        ff = self.feed_forward(x)
        N_x = ff.reshape(*x.shape[:-1],
                                     self.layer_sizes[-1]//self.input_size,
                                     self.input_size)
        P_x = torch.matmul(N_x.transpose(-2,-1),N_x)
        xPx = -torch.matmul(x_vec.transpose(-2,-1),torch.matmul(P_x,x_vec)).squeeze(-1)
        return xPx*(1+self.eps) + self.eps

