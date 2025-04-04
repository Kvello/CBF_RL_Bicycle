from envs.integrator import SafeDoubleIntegratorEnv
from torchrl import envs
from torchrl.envs.common import EnvBase
from torchrl.envs import step_mdp
from tensordict import TensorDict
from torchrl.envs.transforms import TransformedEnv, StepCounter
import torch
import pytest


@pytest.fixture
def env():
    return SafeDoubleIntegratorEnv(device="cpu",seed=0)

def test_env_creation(env):
    """ Ensure that the environment can be created """
    assert isinstance(env, EnvBase)
    assert env.device == torch.device("cpu")

def test_reset(env):
    """Check that reset() produces a valid TensorDict"""
    td = env.reset()
    
    assert isinstance(td, TensorDict)
    assert set(td.keys()) == {"x1", "x2","done","terminated"}

    # Validate observation spec
    assert env.observation_spec.contains(td), "Reset TensorDict does not match observation spec"

def test_safe_double_integrator_env_spec(env):
    envs.check_env_specs(env)
    
def test_safe_double_integrator_env_reset(env):
    td = env.reset()
    obs_spec = env.observation_spec
    assert obs_spec.contains(td)
    
def test_safe_double_integrator_env_random_action(env):
    td = env.reset()
    action = env.action_spec.rand()
    assert env.action_spec.contains(action)
    td.update({"action":action},clone=True)
    stepped = env.step(td)
    data = envs.step_mdp(stepped)
    assert env.observation_spec.contains(data)

def test_safe_double_integrator_env_multi_step(env):
    env.reset()
    rollout = env.rollout(max_steps=10)
    assert env.observation_spec.contains(rollout)

def test_cost_and_reset(env):
    td = env.reset()
    obs_spec = env.observation_spec
    params = env.parameters
    td["x1"] = torch.tensor(0.0,dtype=torch.float32)
    td["x2"] = params["max_x2"] + 0.01
    td["action"] = params["max_input"].clone().detach()
    stepped = env.step(td)
    assert stepped["x2"] >= params["max_x2"], "The purpose of this test is to move the state\
        outside the safety region. The input {td['action']} did not acheive this."
    assert obs_spec.contains(stepped), "The output of step_mdp is not a valid TensorDict"
    assert stepped["next","reward"] == -1.0, "The cost of the state is not positive"
    assert stepped["next","done"] == True, "The episode should have ended"
    
def test_max_steps(env):
    transformed_env = TransformedEnv(env,StepCounter(max_steps=10,
                                                     truncated_key="truncated",
                                                     step_count_key="step_count",
                                                     update_done=True))
    td = transformed_env.reset()
    td["x1"] = torch.tensor(0.0,dtype=torch.float32)
    td["x2"] = torch.tensor(0.0,dtype=torch.float32)
    for _ in range(10):
        # Input 0 to keep the system in the equilibrium
        td["action"] = torch.tensor(0.0,dtype=torch.float32)
        td = transformed_env.step(td) 
        td = step_mdp(td)
    assert td["terminated"] == False, "The episode should not have terminated"
    assert td["done"] == True, "The episode should have ended"
    assert td["truncated"] == True, "The episode should have been truncated"

def test_batched_safe_double_integrator_env():
    batch_size = 16
    env = SafeDoubleIntegratorEnv(batch_size=batch_size,device="cpu")
    td = env.reset()
    assert td["x1"].shape == (batch_size,)
    assert td["x2"].shape == (batch_size,)
    td = env.rollout(max_steps=1,auto_reset=False,tensordict=td)
    assert td["x1"].shape == (batch_size,1)
    obs_spec = env.observation_spec
    assert obs_spec.contains(td[:,0])
def test_batched_max_step_safe_double_integrator_env():
    batch_size = 16
    env = SafeDoubleIntegratorEnv(batch_size=batch_size,device="cpu")
    transformed_env = TransformedEnv(env,StepCounter(max_steps=10,
                                                     truncated_key="truncated",
                                                     step_count_key="step_count",
                                                     update_done=True))
    td = env.reset()
    td["step_count"] = torch.zeros_like(td["terminated"])
    td = transformed_env.reset(td)
    assert td["x1"].shape == (batch_size,)
    assert td["x2"].shape == (batch_size,)
    td["x1"] = torch.zeros(batch_size,dtype=torch.float32)
    td["x2"] = torch.zeros(batch_size,dtype=torch.float32)
    for _ in range(10):
        # Input 0 to keep the system in the equilibrium
        td["action"] = torch.zeros(batch_size,dtype=torch.float32)
        td = transformed_env.step(td) 
        td = step_mdp(td)
    assert (td["terminated"] == False).all(), "The episode should not have terminated"
    assert td["done"].all(), "The episode should have ended"
    assert td["truncated"].all(), "The episode should have been truncated"