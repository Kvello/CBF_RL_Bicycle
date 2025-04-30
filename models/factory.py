from .value import (
    FfValueFunction,
    QuadraticValueFunction,
    ValueBase,
)
from .policy import (
    Ff,
    PolicyBase,
)
from typing import Type

# TODO: Create registration functionality?
class ValueFactory:
    """Factory for creating different Safety Value Function architectures."""
    architectures = {
        "feedforward": FfValueFunction,
        "quadratic": QuadraticValueFunction,
    }

    @staticmethod
    def create(name: str, **kwargs) -> Type[ValueBase]:
        if name not in ValueFactory.architectures:
            raise ValueError(f"Unknown architecture '{name}'.\
                             Available: {list(ValueFactory.architectures.keys())}")
        return ValueFactory.architectures[name](**kwargs)

class PolicyFactory:
    """Factory for creating different Policy architectures."""
    architectures = {
        "feedforward": Ff,
    }

    @staticmethod
    def create(name: str, **kwargs) -> Type[PolicyBase]:
        if name not in PolicyFactory.architectures:
            raise ValueError(f"Unknown architecture '{name}'.\
                             Available: {list(PolicyFactory.architectures.keys())}")
        return PolicyFactory.architectures[name](**kwargs)