from envs.integrator import MultiObjectiveDoubleIntegratorEnv
from torchrl import envs
from torchrl.envs.common import EnvBase
from torchrl.envs import step_mdp
from tensordict import TensorDict
from torchrl.envs.transforms import TransformedEnv, StepCounter
import torch
import pytest

@pytest.fixture
def env():
    return MultiObjectiveDoubleIntegratorEnv(device="cpu")

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

def test_multi_objective_double_integrator_env_spec(env):
    envs.check_env_specs(env)
    
def test_multi_objective_double_integrator_env_reset(env):
    td = env.reset()
    obs_spec = env.observation_spec
    assert obs_spec.contains(td)
    
def test_multi_objective_double_integrator_env_random_action(env):
    td = env.reset()
    action = env.action_spec.rand()
    assert env.action_spec.contains(action)
    td.update({"action":action},clone=True)
    stepped = env.step(td)
    data = envs.step_mdp(stepped)
    assert env.observation_spec.contains(data)

def test_multi_objective_double_integrator_env_multi_step(env):
    env.reset()
    rollout = env.rollout(max_steps=10)
    assert env.observation_spec.contains(rollout)

def test_reward_values(env):
    td = env.reset()
    obs_spec = env.observation_spec
    x1 = torch.tensor(0.0,dtype=torch.float32)
    params = env.params
    td["x1"] = x1
    x2 = params["max_x2"] + 0.01
    td["x2"] = x2
    u = params["max_input"].clone().detach()
    td["action"] = u
    stepped = env.step(td)
    assert stepped["x2"] >= params["max_x2"], "The purpose of this test is to move the state\
        outside the safety region. The input {td['action']} did not acheive this."
    assert obs_spec.contains(stepped)
    assert stepped["next","r1"] == stepped["next","reward"], "The reward for the primary objective should be the same as the reward"
    assert stepped["next","done"] == True, "The episode should terminate after the state moves outside the safety region"
    dt = params["dt"]
    assert stepped["next","r2"] == MultiObjectiveDoubleIntegratorEnv._secondary_reward_func(x1,x2), "The secondary objective is not being calculated correctly"
    
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

def test_batched_multi_objective_double_integrator_env():
    batch_size = 16
    env = MultiObjectiveDoubleIntegratorEnv(batch_size=batch_size, device="cpu")
    td = env.reset()
    assert td["x1"].shape == (batch_size,)
    assert td["x2"].shape == (batch_size,)
    td = env.rollout(max_steps=1,auto_reset=False,tensordict=td)
    assert td["x1"].shape == (batch_size,1)
    td = td[:,0] # get the first step
    obs_spec = env.observation_spec
    assert obs_spec.contains(td)
def test_batched_max_step_multi_objective_double_integrator_env():
    batch_size = 16
    env = MultiObjectiveDoubleIntegratorEnv(batch_size=16, device="cpu")
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