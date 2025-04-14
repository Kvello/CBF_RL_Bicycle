import torch
from typing import Union
import warnings

class Filter:
    """Base class for filters
    
    """
    def __init__(self):
        pass
    def apply(self, data):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError
    def _check_data(self, data):
        mask = torch.isnan(data) | torch.isinf(data)
        if mask.any():
            warnings.warn("Data contains NaN or Inf values")
        return mask

class FirstOrderFilter(Filter):
    """First order filter
    A base class for first order filters. This class is not meant to be used
    directly, but to be subclassed by other filters. It provides a simple
    interface for implementing first order filters.
    """
    def __init__(self, initial_value:Union[float,torch.Tensor]=0.0):
        self.y = initial_value
        self.initial_value = initial_value
        super().__init__()
    def _forward(self, data:torch.Tensor,y:torch.Tensor)->torch.Tensor:
        raise NotImplementedError
    def _output(self, data:torch.Tensor,y:torch.Tensor)->torch.Tensor:
        raise NotImplementedError
    def apply(self, data:torch.Tensor)->torch.Tensor: 
        mask = self._check_data(data)
        if mask.any():
            data[mask] = self.y
        self.y = self._forward(data,self.y)
        return self._output(data,self.y)
    def reset(self):
        self.y = self.initial_value

class LowPassFilter(FirstOrderFilter):
    """Low pass filter
    A standard, first order IIR low pass filter implementation
    y[n] = (1-alpha) * y[n-1] + alpha * x[n]
    """
    def __init__(self, alpha:float, initial_value:Union[float,torch.Tensor]=0.0):
        assert 0 < alpha < 1,"Alpha must be between 0 and 1"
        self.alpha = alpha
        super().__init__(initial_value)
    def _forward(self, data:torch.Tensor,y:torch.Tensor)->torch.Tensor:
        return y + self.alpha * (data - y)
    def _output(self, data:torch.Tensor,y:torch.Tensor)->torch.Tensor:
        return y

class HighPassFilter(FirstOrderFilter):
    """High pass filter
    A standard, first order IIR high pass filter implementation
    y[n] = alpha * y[n-1] + alpha * (x[n] - x[n-1])
    """
    def __init__(self, alpha:float, initial_value:Union[float,torch.Tensor]=0.0):
        assert 0 < alpha < 1,"Alpha must be between 0 and 1"
        self.alpha = alpha
        super().__init__(initial_value)
    def _forward(self, data:torch.Tensor,y:torch.Tensor)->torch.Tensor:
        return y + self.alpha * (data - y)
    def _output(self, data:torch.Tensor,y:torch.Tensor)->torch.Tensor:
        return data - y

class SampleMeanFilter(Filter):
    """Sample mean filter
    A simple filter that computes the sample mean of the data online
    """
    def __init__(self):
        self.n:int = 1
        self.y = 0.0
        super().__init__()
    def apply(self, data:torch.Tensor)->torch.Tensor:
        mask = self._check_data(data)
        if mask.any():
            warnings.warn("Data contains NaN or Inf values")
            data[mask] = self.y
        self.y = self.y + (data - self.y)/self.n
        self.n += 1
        return self.y
    def reset(self):
        self.mean = 0.0
        self.n:int = 1


class RunningNormalizerBase:
    """Base class for running normalizers."""
    def __init__(self, offset: float = 0.0):
        self.offset = offset

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization to the input."""
        raise NotImplementedError

    def reset(self):
        """Reset the internal state."""
        raise NotImplementedError


class RunningMeanNormalizer(RunningNormalizerBase):
    """Normalizer using a running sample mean."""
    def __init__(self, offset: float = 0.0):
        super().__init__(offset)
        self.n = 1
        self.mean = torch.tensor(0.0)
        self.filter = SampleMeanFilter()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.filter.apply(x)
        return x - self.mean + self.offset 

    def reset(self):
        self.filter.reset()

class RunningLowPassNormalizer(RunningNormalizerBase):
    """Normalizer using a first-order low-pass filter."""
    def __init__(self, alpha: float, offset: float = 0.0,initial_value:float=0.0):
        super().__init__(offset)
        self.filter = LowPassFilter(alpha,initial_value)
        self.mean = torch.tensor(0.0)
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.filter.apply(x)
        return x - self.mean + self.offset

    def reset(self):
        self.filter.reset()


class NormalizerFactory:
    """Factory class for creating normalizers."""
    @staticmethod
    def create(normalizer_type: str, **kwargs) -> RunningNormalizerBase:
        normalizers = {
            "mean": RunningMeanNormalizer,
            "lowpass": RunningLowPassNormalizer,
        }
        if normalizer_type not in normalizers:
            raise ValueError(f"Unknown normalizer type: {normalizer_type}")
        return normalizers[normalizer_type](**kwargs)