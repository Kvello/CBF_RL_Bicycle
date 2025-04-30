from abc import ABC, abstractmethod
from tensordict.nn import TensorDictModule
from torchrl.envs import BaseEnv
class BaseRunner():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.env = None
        self.policy_module = None
        self.CDF_module = None
        self.value_module = None
    @abstractmethod
    def setup():
        pass
    @abstractmethod
    def train():
        pass
    @abstractmethod
    def evaluate():
        pass
    @abstractmethod
    def save():
        pass
    @abstractmethod
    def load():
        pass