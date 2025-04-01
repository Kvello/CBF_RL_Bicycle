import pytest
import torch
from utils.filters import LowPassFilter, HighPassFilter, SampleMeanFilter  # Adjust import path accordingly

def test_low_pass_filter():
    alpha = 0.5
    initial_value = torch.tensor(0.0)
    lpf = LowPassFilter(alpha, initial_value)
    
    # Check impuse response
    data = torch.tensor([1.0, 0.0,0.0,0.0])
    output = []
    for d in data:
        output.append(lpf.apply(d).item())
    
    expected = [alpha*(1-alpha)**i for i in range(len(data))]
    assert torch.allclose(torch.tensor(output), torch.tensor(expected), atol=1e-5)

def test_low_pass_filter_batch():
    alpha = 0.5
    initial_value = torch.tensor(0.0)
    lpf = LowPassFilter(alpha, initial_value)
    
    data = torch.tensor([1.0,2.0,3.0,4.0])
    output = lpf.apply(data)
    assert output.shape == data.shape
    # No data should be repeated(filtering done indpendently)
    assert output.numel() == torch.unique(output).numel()

def test_high_pass_filter():
    alpha = 0.5
    initial_value = torch.tensor(0.0)
    hpf = HighPassFilter(alpha, initial_value)
    
    # Check impuse response
    data = torch.tensor([1.0,0.0,0.0,0.0])
    output = []
    for d in data:
        output.append(hpf.apply(d).item())
    
    expected = [(1-alpha)*((1-alpha)**i - (i>0)*(1-alpha)**(i-1)) for i in range(len(data))]
    assert torch.allclose(torch.tensor(output), torch.tensor(expected), atol=1e-5)

def test_high_pass_filter_batch():
    alpha = 0.5
    initial_value = torch.tensor(0.0)
    hpf = HighPassFilter(alpha, initial_value)
    
    data = torch.tensor([1.0,2.0,3.0,4.0])
    output = hpf.apply(data)
    assert output.shape == data.shape
    # No data should be repeated(filtering done indpendently)
    assert output.numel() == torch.unique(output).numel()


def test_sample_mean_filter():
    smf = SampleMeanFilter()
    data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    output = []
    for d in data:
        output.append(smf.apply(d).item())
    
    expected = [sum(data[:i+1])/(i+1) for i in range(len(data))]
    assert torch.allclose(torch.tensor(output), torch.tensor(expected), atol=1e-5)

def test_sample_mean_filter_batch():
    smf = SampleMeanFilter()
    data = torch.tensor([1.0,2.0,3.0,4.0])
    output = smf.apply(data)
    assert output.shape == data.shape
    # No data should be repeated(filtering done indpendently)
    assert output.numel() == torch.unique(output).numel()


def test_reset():
    lpf = LowPassFilter(0.5, torch.tensor(5.0))
    lpf.apply(torch.tensor(10.0))
    lpf.reset()
    assert lpf.y == 5.0
    
    hpf = HighPassFilter(0.5, torch.tensor(5.0))
    hpf.apply(torch.tensor(10.0))
    hpf.reset()
    assert hpf.y == 5.0
    
    smf = SampleMeanFilter()
    smf.apply(torch.tensor(10.0))
    smf.reset()
    assert smf.mean == 0.0 and smf.n == 1

def test_edge_case():
    lpf = LowPassFilter(0.5, torch.tensor(5.0))
    with pytest.warns(UserWarning):
        lpf.apply(torch.tensor(float('nan')))
    assert lpf.y == 5.0
