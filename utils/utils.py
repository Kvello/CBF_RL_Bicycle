from tensordict import TensorDict
from torchrl.envs import EnvBase
from typing import Dict, Any, Optional
import warnings

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