from tensordict import TensorDict
from torchrl.collectors import DataCollectorBase
from torch.optim import Optimizer
from typing import Dict, Any

class RLAlgoBase:
    def __init__(self):
        pass

    def train(self):
        raise NotImplementedError
    def evaluate(self):
        raise NotImplementedError

    def save(self, filepath):
        raise NotImplementedError

    def load(self, filepath):
        raise NotImplementedError

    def _step(self):
        raise NotImplementedError