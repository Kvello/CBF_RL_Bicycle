from tensordict import TensorDict
from torchrl.envs import EnvBase
from typing import Dict, Any, Optional, Callable
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
