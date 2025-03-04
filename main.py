from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from typing import List
from tensordict.nn import TensorDictModule
from torch import nn
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
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
    ParallelEnv,
    EnvCreator,
    EnvBase,
    BatchSizeTransform
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from envs.integrator import SafeDoubleIntegratorEnv, plot_integrator_trajectories, plot_value_function_integrator
from datetime import datetime
import argparse
from results.evaluate import evaluate_policy

class SafetyValueFunction(nn.Module):
    """
    A simple feedforward neural network that represents the value function that is
    supposed to encode safety. The safety-preserving task structure should be used with this 
    value function. 
    This means the value should lie between -1 and 0 for all states.
    Several NN-parameterizations are possible, but the default is a 2-layer feedforward network
    with tanh activations.
    It is reasonable to suspect that the choice of parameterization will affect the
    learned CBF/value function significantly.
    """
    def __init__(self,
                 input_size:int,
                 device:torch.device=torch.device("cpu"),
                 layers:List[int] = [64,64],
                 activation:nn.Module = nn.Tanh()):
        super(SafetyValueFunction, self).__init__()
        self.layers = nn.Sequential() 
        dims = [input_size] + layers + [1]
        for i in range(len(dims)-1):
            self.layers.add_module(f"layer_{i}",nn.Linear(dims[i],dims[i+1],device=device))
            self.layers.add_module(f"activation_{i}",activation)
    def forward(self, x:torch.Tensor):
        return self.layers(x)
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="PPO for Safe Double Integrator")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--load_policy", type=str, default=None, help="Path to load policy")
    parser.add_argument("--load_value", type=str, default=None, help="Path to load value")
    parser.add_argument("--train", action="store_true", default=False, help="Train the model")
    parser.add_argument("--eval", action="store_true", default=False, help="Evaluate the model, both during and after training") 
    parser.add_argument("--plot_traj", type=int, default=0, help="Number of trajectories to plot")
    parser.add_argument("--save", action="store_true", default=False, help="Save the models")
    parser.add_argument("--plot_value", action="store_true", default=False, help="Plot the value function landscape")
    parser.add_argument("--plot_training", action="store_true", default=False, help="Plot training statistics")
    parser.add_argument("--max_rollout_len", type=int, default=100, help="Maximum rollout length")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args() 
    #######################
    # Hyperparameters:
    #######################
    num_cells = 64
    max_input = 1.0
    max_x1 = 1.0
    max_x2 = 1.0
    dt = 0.01
    frames_per_batch = int(2**12)
    lr = 3e-4
    max_grad_norm = 1.0
    total_frames = int(2**22)
    batches_per_process = 16
    num_workers = 8
    sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 10  # optimization steps per batch of data collected
    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    lmbda = 0.95
    entropy_eps = 1e-4
    #######################
    # Environment:
    #######################
    max_rollout_len = args.max_rollout_len
    base_env = SafeDoubleIntegratorEnv(device=device)
    base_env
    env = TransformedEnv(
        base_env,
        Compose(
            BatchSizeTransform([batches_per_process]),
            UnsqueezeTransform(in_keys=["x1", "x2"], dim=-1,in_keys_inv=["x1","x2"]),
            CatTensors(in_keys =["x1", "x2"], out_key= "obs",del_keys=False,dim=-1),
            ObservationNorm(in_keys=["obs"], out_keys=["obs"]),
            DoubleToFloat(),
            StepCounter(max_steps=max_rollout_len),
        )
    )
    gamma = 0.97

    # Handle both batch-locked and unbatched action specs
    action_high = (env.action_spec_unbatched.high if hasattr(env, "action_spec_unbatched") 
                                                    else env.action_spec.high)
    action_low = (env.action_spec_unbatched.low if hasattr(env, "action_spec_unbatched") 
                                                    else env.action_spec.low)
    observation_size_unbatched = (env.obs_size_unbatched if hasattr(env, "obs_size_unbatched") 
                                                            else env.observation_space.shape[0])

    action_size_unbatched = (env.action_size_unbatched if hasattr(env, "action_size_unbatched") 
                                                        else env.action_spec.shape[0])
    env.transform[2].init_stats(num_iter=1000,reduce_dim=0,cat_dim=0)
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
    if args.load_policy is not None:
        policy_module.load_state_dict(torch.load(args.load_policy))
        print("Policy loaded")

    value_net = SafetyValueFunction(observation_size_unbatched,device=device)

    value_module = ValueOperator(
        module=value_net,
        in_keys=["obs"],
    )
    if args.load_value is not None:
        value_module.load_state_dict(torch.load(args.load_value))
        print("Value function loaded") 


    if args.train:
        # Training
        env_creator = EnvCreator(create_batched_env(batch_size=batches_per_process, env=env))
        create_env_fn = [env_creator for _ in range(num_workers)]
        collector = MultiSyncDataCollector(
            create_env_fn=create_env_fn,
            policy=policy_module,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            split_trajs=False,
            device=device,
            exploration_type=ExplorationType.RANDOM)

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
        print("Training")
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
            if i % 10 == 0 and args.eval:
                eval_logs, eval_str = evaluate_policy(env, policy_module, max_rollout_len)
                for key, val in eval_logs.items():
                    logs[key].append(val)
            # We're also using a learning rate scheduler. Like the gradient clipping,
            # this is a nice-to-have but nothing necessary for PPO to work.
            pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
            scheduler.step()
    if args.eval:
        print("Evaluation")
        eval_logs, eval_str = evaluate_policy(env, policy_module, max_rollout_len)
        print(eval_str)
    if args.plot_traj > 0:
        print("Plotted trajectories")
        plot_integrator_trajectories(env, 
                                    policy_module,
                                    max_rollout_len,
                                    args.plot_traj,
                                    max_x1,
                                    max_x2)
    if args.save:
        print("Saving")
        # Save the model
        torch.save(policy_module.state_dict(), "models/ppo_policy_safe_integrator" \
            + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pth")
        torch.save(value_module.state_dict(), "models/ppo_value_safe_integrator" \
            + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pth")
    if args.plot_value:
        print("Plotting value function")
        plot_value_function_integrator(max_x1, max_x2, 10, value_module)
    if args.plot_training:
        # Plot training statistics
        print("Plotting training statistics")
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
        plt.savefig("ppo_safe_integrator_statistics.png")

    # # Plot value-function landscape
    # plt.figure(figsize=(10, 10))
    # resolution = 10 # point per unit
    # x_points = int(2*max_x1*1.1*resolution)
    # y_points = int(2*max_x2*1.1*resolution)
    # x = torch.linspace(-max_x1*1.1, max_x1*1.1, x_points)
    # y = torch.linspace(-max_x2*1.1, max_x2*1.1, y_points)
    # X, Y = torch.meshgrid(x, y, indexing="xy")
    # Z = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    # V = value_net(Z)
    # # CBF is supposed to be -V
    # V = -V.reshape(x_points, y_points)
    # plt.contourf(X, Y, V.detach().numpy(), 20)
    # # Plot lines corresponding to the safe set
    # plt.plot([-max_x1, -max_x1], [-max_x2, max_x2], "r")
    # plt.plot([max_x1, max_x1], [-max_x2, max_x2], "r")
    # plt.plot([-max_x1, max_x1], [-max_x2, -max_x2], "r")
    # plt.plot([-max_x1, max_x1], [max_x2, max_x2], "r")
    # plt.savefig("ppo_safe_integrator_value_function.png")
