from tensordict import TensorDict
from torchrl.envs import EnvBase

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