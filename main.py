from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from torch import nn
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    CatTensors,
    UnsqueezeTransform,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from envs.integrator import SafeDoubleIntegratorEnv, MergeFieldsTransform


device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

#Environment:
max_rollout_len = 100
base_env = SafeDoubleIntegratorEnv(device=device)
env = TransformedEnv(
    base_env,
    Compose(
        UnsqueezeTransform(in_keys=["x1", "x2"], dim=-1,in_keys_inv=["x1","x2"]),
        CatTensors(in_keys =["x1", "x2"], out_key= "obs",del_keys=False,dim=-1),
        ObservationNorm(in_keys=["obs"], out_keys=["obs"]),
        DoubleToFloat(),
        StepCounter(max_steps=max_rollout_len),
    )
)
env.transform[2].init_stats(num_iter=1000,reduce_dim=0,cat_dim=0)
# Hyperparameters:
num_cells = 64
frames_per_batch = 1000
lr = 3e-4
max_grad_norm = 1.0
total_frames = 1_000_000
sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4
# Handle both batch-locked and unbatched action specs
action_high = (env.action_spec_unbatched.high if hasattr(env, "action_spec_unbatched") 
                                                else env.action_spec.high)
action_low = (env.action_spec_unbatched.low if hasattr(env, "action_spec_unbatched") 
                                                else env.action_spec.low)
observation_size_unbatched = (env.obs_size_unbatched if hasattr(env, "obs_size_unbatched") 
                                                        else env.observation_space.shape[0])

action_size_unbatched = (env.action_size_unbatched if hasattr(env, "action_size_unbatched") 
                                                        else env.action_spec.shape[0])
actor_net = nn.Sequential(
    nn.Linear(observation_size_unbatched, num_cells,device=device),
    nn.Tanh(),
    nn.Linear(num_cells, num_cells,device=device),
    nn.Tanh(),
    nn.Linear(num_cells, num_cells,device=device),
    nn.Tanh(),
    nn.Linear(num_cells,2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)



policy_module = ProbabilisticActor(
    module=TensorDictModule(actor_net,
                            in_keys=["obs"],
                            out_keys=["loc", "scale"]),
    in_keys=["loc", "scale"],
    spec=env.action_spec,
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": action_low,
        "high": action_high,
    },
    return_log_prob=True,
)


value_net = nn.Sequential(
    nn.Linear(observation_size_unbatched, num_cells,device=device),
    nn.Tanh(),
    nn.Linear(num_cells, num_cells,device=device),
    nn.Tanh(),
    nn.Linear(num_cells, num_cells,device=device),
    nn.Tanh(),
    nn.Linear(num_cells,1,device=device)
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["obs"],
)

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=gamma,
    lmbda=lmbda,
    value_network=value_module,
    average_gae=True
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

# Training
logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(logs["reward"])
plt.title("training rewards (average)")
plt.subplot(2, 2, 2)
plt.plot(logs["step_count"])
plt.title("Max step count (training)")
plt.subplot(2, 2, 3)
plt.plot(logs["eval reward (sum)"])
plt.title("Return (test)")
plt.subplot(2, 2, 4)
plt.plot(logs["eval step_count"])
plt.title("Max step count (test)")
plt.show()