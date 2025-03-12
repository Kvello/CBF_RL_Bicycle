from .value import (
    FfSafetyValueFunction,
    QuadraticSafetyValueFunction,
    SafetyValueFunctionBase,
)
from typing import Type

# TODO: Create registration functionality?
class SafetyValueFunctionFactory:
    """Factory for creating different Safety Value Function architectures."""
    architectures = {
        "feedforward": FfSafetyValueFunction,
        "quadratic": QuadraticSafetyValueFunction,
    }

    @staticmethod
    def create(name: str, **kwargs) -> Type[SafetyValueFunctionBase]:
        if name not in SafetyValueFunctionFactory.architectures:
            raise ValueError(f"Unknown architecture '{name}'.\
                             Available: {list(SafetyValueFunctionFactory.architectures.keys())}")
        return SafetyValueFunctionFactory.architectures[name](**kwargs)