from tensordict import TensorDict
from torchrl.envs import EnvBase
from typing import Dict, Any, Optional
import warnings
import torch

def get_config_value(config:Dict[str,Any], 
                     key:str, 
                     default:Any,
                     warn_messsage:Optional[str]=None):
    """
    Get a value from a config dictionary, if it is not present
    return the default value.
    """
    try:
        return config[key]
    except KeyError:
        if warn_messsage is not None:
            warnings.warn(warn_messsage)
        return default

def gradient_projection(
    common_module:torch.nn.Module,
    primary_loss:torch.Tensor,
    secondary_loss:torch.Tensor)->torch.Tensor:
        """ Calculates a projected gradient of the secondary loss onto the nullspace
        of the primary loss. Returns the sum of the primary loss gradient and the
        projected secondary loss gradient.

        Args:
            common_module (torch.nn.Module): Common module with which the two losses were computed 
            containing the parameters
            primary_loss (torch.Tensor): The primary loss tensor
            secondary_loss (torch.Tensor): The secondary loss tensor

        Returns:
            torch.Tensor: The projected and combined gradient
            
        Note:
            Calling this funciton clears the gradients of the common module,
            and does not retain the computational graph after the projection.
            This means that to recompute the gradients, the losses have to be recomputed as well.
        """
        # Primary objective loss gradient
        primary_loss.backward(retain_graph=True)
        grad_vec_primary_loss = torch.cat(
            [p.grad.view(-1) for p in common_module.parameters()]
        ) 
        common_module.zero_grad()
        # Secondary objective loss gradient
        secondary_loss.backward()
        grad_vec_secondary_loss = torch.cat(
            [p.grad.view(-1) for p in common_module.parameters()]
        )
        common_module.zero_grad()
        # Projection of secondary objective loss gradient onto nullspace of
        # primary objective loss gradient
        if grad_vec_primary_loss.norm() == 0:
            secondary_proj = torch.zeros_like(grad_vec_secondary_loss)
        else:
            secondary_proj =(
                torch.dot(grad_vec_secondary_loss, grad_vec_primary_loss)
                / torch.dot(grad_vec_primary_loss, grad_vec_primary_loss)
                * grad_vec_primary_loss
            )

        grad = (
            grad_vec_secondary_loss - secondary_proj + grad_vec_primary_loss
        )
        return grad