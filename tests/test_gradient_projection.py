from utils.utils import gradient_projection
from torch import nn
import torch
import pytest

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1, bias=False)  # Simple 2D linear layer

    def forward(self, x):
        return self.fc(x)

@pytest.fixture
def model_and_data():
    """Fixture to create a simple model and data."""
    model = SimpleModel()
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    target1 = torch.tensor([[1.0]])
    target2 = torch.tensor([[2.0]])
    loss_fn = nn.MSELoss()
    return model, x, target1, target2, loss_fn

def test_gradient_projection_shape(model_and_data):
    """Test if the function runs and returns the correct shape."""
    model, x, target1, target2, loss_fn = model_and_data
    pred = model(x)
    primary_loss = loss_fn(pred, target1)
    secondary_loss = loss_fn(pred, target2)

    projected_grad = gradient_projection(model, primary_loss, secondary_loss)

    assert projected_grad is not None, "Projected gradient is None"
    assert projected_grad.shape == torch.Size([2]), "Incorrect gradient shape"

def test_projection_orthogonality(model_and_data):
    """Test if the projected secondary gradient is orthogonal to the primary gradient."""
    model, x, target1, target2, loss_fn = model_and_data
    pred = model(x)
    primary_loss = loss_fn(pred, target1)
    secondary_loss = loss_fn(pred, target2)

    projected_grad = gradient_projection(model, primary_loss, secondary_loss)

    # Compute primary gradient manually
    # Graph is cleared after projection, so we need to compute the primary gradient again
    pred = model(x)
    primary_loss = loss_fn(pred, target1)
    primary_loss.backward()
    primary_grad = torch.cat([p.grad.view(-1) for p in model.parameters()])
    model.zero_grad()

    # Check orthogonality: dot product should be ~0
    dot_product = torch.dot(projected_grad - primary_grad, primary_grad)
    assert torch.isclose(dot_product, torch.tensor(0.0), atol=1e-4), "Secondary gradient is not orthogonal to primary gradient"

def test_primary_gradient_preserved(model_and_data):
    """Test if the primary gradient component is retained in the final projected gradient."""
    model, x, target1, target2, loss_fn = model_and_data
    pred = model(x)
    primary_loss = loss_fn(pred, target1)
    secondary_loss = loss_fn(pred, target2)

    projected_grad = gradient_projection(model, primary_loss, secondary_loss)

    # Compute primary gradient manually
    # Graph is cleared after projection, so we need to compute the primary gradient again
    pred = model(x)
    primary_loss = loss_fn(pred, target1)
    primary_loss.backward(retain_graph=True)
    primary_grad = torch.cat([p.grad.view(-1) for p in model.parameters()])
    model.zero_grad()

    # Compute the component of projected_grad along primary_grad
    primary_grad_norm_sq = torch.dot(primary_grad, primary_grad)
    projected_primary_component = (
        torch.dot(projected_grad, primary_grad) / primary_grad_norm_sq
    ) * primary_grad
    # The projected gradient should contain the primary gradient component
    assert torch.allclose(projected_primary_component, primary_grad, atol=1e-6), \
        "The primary gradient component is not fully retained."


@pytest.mark.parametrize("target1, target2", [
    (torch.tensor([[1.0]]), torch.tensor([[1.0]])),  # Identical losses
    (torch.tensor([[0.0]]), torch.tensor([[0.0]])),  # Zero gradients
])
def test_edge_cases(model_and_data, target1, target2):
    """Test edge cases where losses are identical or produce zero gradients."""
    model, x, _, _, loss_fn = model_and_data  # Ignore the default targets
    pred = model(x)
    primary_loss = loss_fn(pred, target1)
    secondary_loss = loss_fn(pred, target2)

    projected_grad = gradient_projection(model, primary_loss, secondary_loss)

    # Compute primary and secondary gradients manually
    # Graph is cleared after projection, so we need to compute the gradients again
    pred = model(x)
    primary_loss = loss_fn(pred, target1)
    secondary_loss = loss_fn(pred, target2)
    primary_loss.backward(retain_graph=True)
    primary_grad = torch.cat([p.grad.view(-1) for p in model.parameters()])
    model.zero_grad()
    secondary_loss.backward()
    secondary_grad = torch.cat([p.grad.view(-1) for p in model.parameters()])
    model.zero_grad()

    assert projected_grad is not None, "Projected gradient should not be None"
    assert torch.isnan(projected_grad).sum() == 0, "Projected gradient should not contain NaNs"
    assert torch.isinf(projected_grad).sum() == 0, "Projected gradient should not contain infinities"
    assert projected_grad.shape == torch.Size([2]), "Incorrect gradient shape"
    assert torch.allclose(projected_grad, primary_grad, atol=1e-6), \
        "The projected gradient should be equal to the primary gradient"
    assert torch.allclose(projected_grad, secondary_grad, atol=1e-6), \
        "The projected gradient should be equal to the secondary gradient"
        

