import torch
import torch.nn as nn
import pytest
from typing import List
from models.cdf import (
    FfCDF, 
    QuadraticCDF, 
)
from models.factory import CDFFactory

def test_ff_safety_value_function_shapes():
    input_size = 10
    batch_size = 32
    model = FfCDF(input_size=input_size, 
                                  layers=[64, 64], 
                                  activation=nn.ReLU(), 
                                  bounded=True)
    
    x = torch.randn(batch_size, input_size)
    output = model(x)
    
    assert output.shape == (batch_size, 1), "Feedforward network output shape incorrect"
    assert output.min().item() >= -1 and output.max().item() <= 0, "Bounded output should \
        be in range [-1, 0]"

def test_ff_safety_value_function_unbounded():
    input_size = 10
    batch_size = 5
    model = FfCDF(input_size=input_size, layers=[64, 64], activation=nn.ReLU(), bounded=False)
    x = torch.randn(batch_size, input_size)
    output = model(x)
    assert output.shape == (batch_size, 1), "Unbounded feedforward network \
        output shape incorrect"

def test_quadratic_safety_value_function_shapes():
    input_size = 10
    batch_size = 32
    layers = [20, 20]  # Ensure last layer is a multiple of input_size
    model = QuadraticCDF(input_size=input_size, 
                                         layers=layers, 
                                         activation=nn.ReLU())
    
    x = torch.randn(batch_size, input_size)
    output = model(x)
    
    assert output.shape == (batch_size,1), "Quadratic network output shape incorrect"

def test_quadratic_safety_value_function_psd():
    input_size = 5
    batch_size = 5
    model = QuadraticCDF(input_size=input_size, 
                                         layers=[10], 
                                         activation=nn.ReLU(),
                                         eps=0.0)
    
    x = torch.ones(batch_size, input_size)
    output = model(x)
    
    assert torch.all(output <= 0), "Quadratic form should be non-positive \
        (P(x) should be PSD)"

def test_factory_creation():
    input_size = 10
    layers = [64, 64,100]
    activation = nn.ReLU()
    device = torch.device("cpu")
    config_ff = {
        "input_size": input_size,
        "layers": layers,
        "activation": activation,
        "device": device,
        "bounded": True
    } 
    ff_model = CDFFactory.create("feedforward",
                                                    **config_ff)
    assert isinstance(ff_model, FfCDF), "Factory did not create\
        correct feedforward model"

    config_quad = {
        "input_size": input_size,
        "layers": layers,
        "activation": activation,
        "device": device
    }
    quad_model = CDFFactory.create("quadratic",
                                                    **config_quad)

    assert isinstance(quad_model, QuadraticCDF), "Factory did not \
        create correct quadratic model"

def test_factory_invalid_architecture():
    config_invalid = {}
    with pytest.raises(ValueError, match="Unknown architecture"):
        CDFFactory.create("invalid_arch",
                                            **config_invalid)

if __name__ == "__main__":
    pytest.main()
