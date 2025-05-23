from envs.integrator import DoubleIntegratorEnv
from torchrl import envs
from torchrl.envs.common import EnvBase
from torchrl.envs import step_mdp
from tensordict import TensorDict
from torchrl.envs.transforms import TransformedEnv, StepCounter
import torch
import pytest


@pytest.fixture
def env():
    return DoubleIntegratorEnv(device="cpu",seed=0)

def test_env_creation(env):
    """ Ensure that the environment can be created """
    assert isinstance(env, EnvBase)
    assert env.device == torch.device("cpu")

def test_reset(env):
    """Check that reset() produces a valid TensorDict"""
    td = env.reset()
    
    assert isinstance(td, TensorDict)
    assert set(td.keys()) == {"x1", "x2","done","terminated","reference_index","x1_ref","x2_ref"}

    # Validate observation spec
    assert env.observation_spec.contains(td), "Reset TensorDict does not match observation spec"

def test_double_integrator_env_spec(env):
    envs.check_env_specs(env)
    
def test_double_integrator_env_reset(env):
    td = env.reset()
    obs_spec = env.observation_spec
    assert obs_spec.contains(td)
    
def test_double_integrator_env_random_action(env):
    td = env.reset()
    action = env.action_spec.rand()
    assert env.action_spec.contains(action)
    td.update({"action":action},clone=True)
    stepped = env.step(td)
    data = envs.step_mdp(stepped)
    assert env.observation_spec.contains(data)

def test_reward_values(env):
    td = env.reset()
    obs_spec = env.observation_spec
    x1 = torch.tensor(0.0,dtype=torch.float32)
    params = env.params
    td["x1"] = x1
    td["reference_index"] = torch.tensor(0,dtype=torch.int32)
    td["x1_ref"] = torch.tensor(0.0,dtype=torch.float32)
    x2 = params["max_x2"] + 0.01
    td["x2"] = x2
    td["x2_ref"] = x2
    u = params["max_input"].clone().detach()
    td["action"] = u
    stepped = env.step(td)
    assert stepped["x2"] >= params["max_x2"], "The purpose of this test is to move the state\
        outside the safety region. The input {td['action']} did not acheive this."
    assert obs_spec.contains(stepped)
    assert stepped["next","done"] == True, "The episode should terminate after the state moves outside the safety region"
    assert stepped["next",DoubleIntegratorEnv.secondary_reward_key] == 0.0, "The reward for the secondary objective should be zero"
    td["x1"] = torch.tensor(0.0,dtype=torch.float32)
    td["x2"] = torch.tensor(0.0,dtype=torch.float32)
    td["action"] = torch.tensor(0.0,dtype=torch.float32)
    td["reference_index"] = torch.tensor(0,dtype=torch.int32)
    
    dt = params["dt"]
    A_ref = params["reference_amplitude"].clone().detach()
    f_ref = params["reference_frequency"].clone().detach()

    td["x1_ref"] = torch.tensor(0.0,dtype=torch.float32)
    td["x2_ref"] = A_ref*2*torch.pi*f_ref
    for n in range(10):
        # Input 0 to keep the system in the equilibrium
        td = env.step(td) 
        x1_ref = A_ref * torch.sin(f_ref * n*2*torch.pi*dt)
        x2_ref = A_ref*2*torch.pi*f_ref*torch.cos(f_ref * n*2*torch.pi*dt)
        Y = torch.stack([x1_ref,x2_ref],dim=0)
        assert td["next",DoubleIntegratorEnv.primary_reward_key] == 0.0, "The reward for the primary objective should be zero"
        assert torch.allclose(-td["next",DoubleIntegratorEnv.secondary_reward_key],torch.linalg.vector_norm(Y,dim=0,ord=2),atol=1e-6)
        td = step_mdp(td)
        td["action"] = torch.tensor(0.0,dtype=torch.float32)
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
    env = DoubleIntegratorEnv(batch_size=batch_size,device="cpu")
    td = env.reset()
    assert td["x1"].shape == (batch_size,)
    assert td["x2"].shape == (batch_size,)
    td = env.rollout(max_steps=1,auto_reset=False,tensordict=td)
    assert td["x1"].shape == (batch_size,1)
    obs_spec = env.observation_spec
    assert obs_spec.contains(td[:,0])
def test_batched_max_step_safe_double_integrator_env():
    batch_size = 16
    env = DoubleIntegratorEnv(batch_size=batch_size,device="cpu")
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
    assert td["truncated"].all(), "The episode should have been truncated"@pytest.fixture
