from envs.integrator import SafeDoubleIntegratorEnv
from torchrl import envs
from torchrl.envs.common import EnvBase
from torchrl.envs import step_mdp
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
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
    params = env.params
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

def test_initial_state_buffer_all_colisions():
    batch_size = 8
    env = SafeDoubleIntegratorEnv(batch_size=batch_size,
                                  buffer_reset_fraction=1.0)
    buffer = env._initial_state_buffer
    assert buffer is not None, "The initial state buffer should not be None"
    td = env.reset()
    params = env.params
    x1 = torch.linspace(-params["max_x1"],
                              params["max_x1"],
                              batch_size)
    x2 = torch.ones(batch_size) * params["max_x2"] + 0.01
    td["x1"] = x1
    td["x2"] = x2
    td["action"] = torch.ones(batch_size,1)*params["max_input"].clone().detach()
    td = env.step(td)
    assert (td["next","reward"] == -1.0).all(), "The cost of the state is not positive"
    env.extend_initial_state_buffer(td)
    buffer = env._initial_state_buffer
    assert buffer["x1"].shape == (batch_size,)
    assert buffer["x2"].shape == (batch_size,)
    assert torch.equal(
        torch.sort(buffer["x1"]).values,torch.sort(x1).values,
        ), "The initial state buffer should contain the colliding states"
    assert torch.equal(
        torch.sort(buffer["x2"]).values,torch.sort(x2).values,
        ), "The initial state buffer should contain the colliding states"
    td = env.reset()
    assert td["x1"].shape == (batch_size,)
    assert td["x2"].shape == (batch_size,)
    assert torch.equal(
        torch.sort(td["x1"]).values,torch.sort(x1).values,
        ), "The initial state buffer should contain the colliding states"
    assert torch.equal(
        torch.sort(td["x2"]).values,torch.sort(x2).values,
        ), "The initial state buffer should contain the colliding states"

def test_initial_state_buffer_half_collisions():
    batch_size = 8
    buffer_frac = 0.5
    env = SafeDoubleIntegratorEnv(batch_size=batch_size,
                                  buffer_reset_fraction=buffer_frac)
    buffer = env._initial_state_buffer
    assert buffer is not None, "The initial state buffer should not be None"
    td = env.reset()
    params = env.params
    x1_collision = torch.linspace(-params["max_x1"],
                              params["max_x1"],
                              int(batch_size*buffer_frac))
    x2_collision = torch.ones(int(batch_size*buffer_frac)) * params["max_x2"] + 0.01
    x1_safe = torch.linspace(-params["max_x1"],
                              params["max_x1"],
                              int(batch_size*buffer_frac))
    x2_safe = torch.zeros(int(batch_size*buffer_frac))
    td["x1"] = torch.cat([x1_collision,x1_safe])
    td["x2"] = torch.cat([x2_collision,x2_safe])
    td["action"] = torch.zeros(batch_size,1)
    td = env.step(td)
    assert td["next","reward"].sum()==pytest.approx(-1*buffer_frac*batch_size,abs=1e-6),\
        "The cost of the state is not positive"
    env.extend_initial_state_buffer(td)
    buffer = env._initial_state_buffer
    assert buffer["x1"].shape == (batch_size*buffer_frac,)
    assert buffer["x2"].shape == (batch_size*buffer_frac,)
    assert torch.isin(x1_collision,buffer["x1"]).all(), "All the collision states should be in the buffer"
    assert torch.isin(x2_collision,buffer["x2"]).all(), "All the collision states should be in the buffer"
    td = env.reset()
    assert td["x1"].shape == (batch_size,)
    assert td["x2"].shape == (batch_size,)
    assert torch.isin(x1_collision,td["x1"]).all(), "All the collision states should be returned from the reset"
    assert torch.isin(x2_collision,td["x2"]).all(), "All the collision states should be returned from the reset" 

def test_initial_state_buffer_multistep():
    batch_size = 8
    buffer_frac = 1.0
    env = SafeDoubleIntegratorEnv(batch_size=batch_size,
                                  buffer_reset_fraction=buffer_frac)
    buffer = env._initial_state_buffer
    assert buffer is not None, "The initial state buffer should not be None"
    td = env.reset()
    params = env.params
    x1 = torch.linspace(-params["max_x1"],
                              params["max_x1"],
                              int(batch_size*buffer_frac))
    x2 = torch.ones(int(batch_size*buffer_frac)) * params["max_x2"] - 0.1
    # All initial states are not safety violating
    td["x1"] = x1
    td["x2"] = x2
    # Apply max force 
    u = torch.ones(batch_size,1)*params["max_input"].clone().detach()
    policy = TensorDictModule(lambda : u,in_keys=[],out_keys=["action"])
    # 11 timesteps should be enough to ensure that all the states are safety violating
    td = env.rollout(max_steps=12,
                     tensordict=td,
                     auto_reset=False,
                     break_when_any_done=False,
                     policy=policy)
    assert td["next","reward"].sum().item() == -1*buffer_frac*batch_size,\
        "The test assumes that all the states should be safety violating"
    env.extend_initial_state_buffer(td)
    buffer = env._initial_state_buffer
    assert buffer["x1"].shape == (batch_size,)
    assert buffer["x2"].shape == (batch_size,)
    assert torch.isin(x1,buffer["x1"]).all(), "All the collision states should be in the buffer"
    assert torch.isin(x2,buffer["x2"]).all(), "All the collision states should be in the buffer"
    td = env.reset()
    assert td["x1"].shape == (batch_size,)
    assert td["x2"].shape == (batch_size,)
    assert torch.isin(x1,td["x1"]).all(), "All the collision states should be returned from the reset"
    assert torch.isin(x2,td["x2"]).all(), "All the collision states should be returned from the reset"
    
