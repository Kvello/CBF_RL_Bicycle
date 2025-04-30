from .cdf import (
    FfCDF,
    QuadraticCDF,
    CDFBase,
)
from typing import Type

# TODO: Create registration functionality?
class CDFFactory:
    """Factory for creating different Safety Value Function architectures."""
    architectures = {
        "feedforward": FfCDF,
        "quadratic": QuadraticCDF,
    }

    @staticmethod
    def create(name: str, **kwargs) -> Type[CDFBase]:
        if name not in CDFFactory.architectures:
            raise ValueError(f"Unknown architecture '{name}'.\
                             Available: {list(CDFFactory.architectures.keys())}")
        return CDFFactory.architectures[name](**kwargs)