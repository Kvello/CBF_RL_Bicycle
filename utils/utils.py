from tensordict import TensorDict
from torchrl.envs import EnvBase
from typing import Dict, Any, Optional, Callable
import warnings
import torch

def reset_batched_env(td:TensorDict, 
                      td_reset:TensorDict, 
                      env:EnvBase):
    """
    Generic reset funciton for non-batched locked env
    batched with BatchTransform.
    """
    td = env.gen_params([*env.batch_size])
    td_new = env.base_env.reset(td)
    return td_new

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
    secondary_loss:torch.Tensor,
    gradient_scaler:Callable[float,float]= lambda x: x)->torch.Tensor:
        """ Calculates a projected gradient of the secondary loss onto the nullspace
        of the primary loss. Returns the sum of the primary loss gradient and the
        projected secondary loss gradient. I.e
        .. math::
            \begin{align}
                \boldsymbol{\eta} &= \nabla_{\theta} L_2^{\text{CLIP}}(\theta) - \text{Proj}_{\nabla_{\theta} L_1^{\text{CLIP}}(\theta)}(\nabla_{\theta} L_2^{\text{CLIP}}(\theta))\\
                \boldsymbol{\delta_\pi} &= \nabla_{\theta} L_1^{\text{CLIP}}(\theta) + w^{\text{CLIP}}\boldsymbol{\eta}
                w^{\text{CLIP}} &= l_r \cdot \frac{||\nabla_{\theta} L_1^{\text{CLIP}}(\theta)||}{||\boldsymbol{\eta}||}
            \end{align}
        Where l_r is the output of the gradient_scaler function, and becomes the relative length
        between the primary and projected secondary loss gradients. Theta are the parameters of the
        common_module
        
        Args:
            common_module (torch.nn.Module): Common module with which the two losses were computed 
            containing the parameters
            primary_loss (torch.Tensor): The primary loss tensor
            secondary_loss (torch.Tensor): The secondary loss tensor
            gradient_scaler (Callable[float,float], optional): A function that scales the secondary
            loss gradient with respect to the primary loss gradient. This will typically be a filter,
            or a constant. Defaults to lambda x: x. Helps with stability.
        Returns:
            torch.Tensor: The projected and combined gradient
            
        Note:
            Calling this funciton clears the gradients of the common module,
            and does not retain the computational graph after the projection.
            This means that to recompute the gradients, the losses have to be recomputed
            as well.
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
        if torch.isclose(grad_vec_primary_loss.norm(),torch.tensor(0.0),atol=1e-30):
            # If primary loss gradient is zero, return secondary loss gradient
            # This is an edge case, and should not happen in practice
            return grad_vec_secondary_loss
        # Projection of secondary objective loss gradient onto nullspace of
        # primary objective loss gradient
        secondary_proj =(
            torch.dot(grad_vec_secondary_loss, grad_vec_primary_loss)
            / torch.dot(grad_vec_primary_loss, grad_vec_primary_loss)
            * grad_vec_primary_loss
        )
        secondary_proj = grad_vec_secondary_loss - secondary_proj 
        relative_length = gradient_scaler(secondary_proj.norm() / grad_vec_primary_loss.norm())
        # Make sure gradient_scaler is ran in the case of a zero secondary loss gradient,
        # as this still is a valid case for the gradient scaler, and should be reported to the
        # gradient scaler if it is statefull(e.g a filter).
        if torch.isclose(secondary_proj.norm(),torch.tensor(0.0),atol=1e-30):
            # If secondary loss gradient is zero, ignore the projection
            # and return the primary loss gradient
            return grad_vec_primary_loss 
        secondary_proj = secondary_proj/secondary_proj.norm()
        grad = (
            secondary_proj*relative_length + grad_vec_primary_loss
        )
        return grad