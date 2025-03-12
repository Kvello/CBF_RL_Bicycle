from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from typing import List
from tensordict.nn import TensorDictModule
from torch import nn
from tensordict.nn.distributions import NormalParamExtractor
from tensordict import TensorDict, TensorDictBase
from torchrl.collectors import MultiSyncDataCollector
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
from torch import multiprocessing
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from envs.integrator import SafeDoubleIntegratorEnv, plot_integrator_trajectories, plot_value_function_integrator
from datetime import datetime
import argparse
from results.evaluate import evaluate_policy, calculate_bellman_violation
from utils.utils import reset_batched_env   
import wandb
from models.factory import SafetyValueFunctionFactory


multiprocessing.set_start_method("spawn", force=True)
is_fork = multiprocessing.get_start_method(allow_none=True) == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="PPO for Safe Double Integrator")
    parser.add_argument("--load_policy", type=str, default=None, help="Path to load policy")
    parser.add_argument("--load_value", type=str, default=None, help="Path to load value")
    parser.add_argument("--train", action="store_true", default=False, help="Train the model")
    parser.add_argument("--eval", action="store_true", default=False, help="Evaluate the model, both during and after training") 
    parser.add_argument("--plot_traj", type=int, default=0, help="Number of trajectories to plot")
    parser.add_argument("--save", action="store_true", default=False, help="Save the models")
    parser.add_argument("--plot_value", action="store_true", default=False, help="Plot the value function landscape")
    parser.add_argument("--plot_training", action="store_true", default=False, help="Plot training statistics")
    parser.add_argument("--max_rollout_len", type=int, default=100, help="Maximum rollout length")
    parser.add_argument("--track", action="store_true", default=False, help="Track the training with wandb")
    parser.add_argument("--wandb_project", type=str, default="ppo_safe_integrator", help="Wandb project name")
    parser.add_argument("--experiment_name", type=str, default=None, help="Wandb experiment name")
    parser.add_argument("--track_bellman_violation", action="store_true", 
                        default=False, help="Track the Bellman violation")
    parser.add_argument("--plot_bellman_violation", action="store_true",
                        default=False, help="Plot the Bellman violation of trained value function and policy")
    
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args() 
    if args.track:
        wandb.init(project=args.wandb_project,
                   sync_tensorboard=True,
                   monitor_gym=True,
                   save_code=True,
                   name=args.experiment_name)

    #######################
    # Hyperparameters:
    #######################
    num_cells = 64
    max_input = 1.0
    max_x1 = 1.0
    max_x2 = 1.0
    dt = 0.05
    frames_per_batch = int(2**16)
    lr = 1e-5
    max_grad_norm = 1.0
    total_frames = int(2**22)
    num_epochs = 20  # optimization steps per batch of data collected
    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    sub_batch_size = int(2**10)
    lmbda = 0.95
    entropy_eps = 0.0
    value_net_config = {
        "name": "feedforward",
        "layers": [64, 64],
        "activation": nn.ReLU(),
        "bounded": True,
        "device": device,
        "input_size": 2,
    }
        
    #######################
    # Parallelization:
    #######################
    if device.type == "cuda":
        batches_per_process = int(2**12)
        num_workers = 1
    else:
        batches_per_process = int(2**12)
        num_workers = 1
    #######################
    # Environment:
    #######################
    state_space = {"x1": {"low": -max_x1, "high": max_x1},
                    "x2": {"low": -max_x2, "high": max_x2}}
    max_rollout_len = args.max_rollout_len
    parameters = TensorDict({
        "params" : TensorDict({
            "dt": dt,
            "max_x1": max_x1,
            "max_x2": max_x2,
            "max_input": max_input,
        },[],
        device=device)
        },[],device=device)
        
    base_env = SafeDoubleIntegratorEnv(device=device,td_params=parameters)
    after_batch_transform = [
            UnsqueezeTransform(in_keys=["x1", "x2"], dim=-1,in_keys_inv=["x1","x2"]),
            CatTensors(in_keys =["x1", "x2"], out_key= "obs",del_keys=False,dim=-1),
            ObservationNorm(in_keys=["obs"], out_keys=["obs"]),
            DoubleToFloat(),
            StepCounter(max_steps=max_rollout_len)]
    env = TransformedEnv(
        base_env,
        Compose(
            BatchSizeTransform(batch_size=[batches_per_process],
                               reset_func=reset_batched_env,
                               env_kwarg=True),
            *after_batch_transform
        )
    ).to(device)
    env.transform[3].init_stats(num_iter=1000,reduce_dim=(0,1),cat_dim=1)
    gamma = 0.99

    
    # Handle both batch-locked and unbatched action specs
    action_high = (env.action_spec_unbatched.high if hasattr(env, "action_spec_unbatched") 
                                                    else env.action_spec.high)
    action_low = (env.action_spec_unbatched.low if hasattr(env, "action_spec_unbatched") 
                                                    else env.action_spec.low)
    observation_size_unbatched = (env.obs_size_unbatched if hasattr(env, "obs_size_unbatched") 
                                                            else env.observation_space.shape[0])

    action_size_unbatched = (env.action_size_unbatched if hasattr(env, "action_size_unbatched") 
                                                        else env.action_spec.shape[0])
    
    #######################
    # Models:
    #######################
    actor_net = nn.Sequential(
        nn.Linear(observation_size_unbatched, num_cells,device=device),
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

    value_net = SafetyValueFunctionFactory.create(**value_net_config)
    value_module = ValueOperator(
        module=value_net,
        in_keys=["obs"],
    )
    if args.load_value is not None:
        value_module.load_state_dict(torch.load(args.load_value))
        print("Value function loaded") 


    #######################
    # Training:
    #######################

    if args.train:
        env_creator = EnvCreator(lambda: env)
        create_env_fn = [env_creator for _ in range(num_workers)]
        collector = MultiSyncDataCollector(
            create_env_fn=create_env_fn,
            policy=policy_module,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            split_trajs=False,
            device=device,
            exploration_type=ExplorationType.RANDOM,
            cat_results=0)

        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=frames_per_batch,device=device),
            sampler=SamplerWithoutReplacement(),
        )

        advantage_module = GAE(
            gamma=gamma,
            lmbda=lmbda,
            value_network=value_module,
            average_gae=True,
            device=device,
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
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optim, total_frames // frames_per_batch, 1e-8
        # )
        print("Training")
        logs = defaultdict(list)
        pbar = tqdm(total=total_frames)
        eval_str = ""

        for i, tensordict_data in enumerate(collector):
            logs["loss_objective"] = 0.0
            logs["loss_critic"] = 0.0
            logs["loss_entropy"] = 0.0
            for _ in range(num_epochs):
                advantage_module(tensordict_data.to(device))
                data_view = tensordict_data.reshape(-1)
                replay_buffer.extend(data_view)
                for _ in range(frames_per_batch // sub_batch_size):
                    subdata = replay_buffer.sample(sub_batch_size).to(device)
                    loss_vals = loss_module(subdata)
                    if entropy_eps == 0.0:
                        loss_vals["loss_entropy"] = torch.tensor(0.0).to(device)
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )
                    logs["loss_objective"] += loss_vals["loss_objective"].item()
                    logs["loss_critic"] += loss_vals["loss_critic"].item()
                    logs["loss_entropy"] += loss_vals["loss_entropy"].item()
                    
                    loss_value.backward()
                    # this is not strictly mandatory but it's good practice to keep
                    # your gradient norm bounded
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                    optim.step()
                    optim.zero_grad()
            logs["loss_objective"] /= num_epochs
            logs["loss_critic"] /= num_epochs
            logs["loss_entropy"] /= num_epochs
            logs["reward"] = tensordict_data["next", "reward"].mean().item()
            pbar.update(tensordict_data.numel())
            cum_reward_str = (
                f"average reward={logs['reward']: 4.4f}"
            )
            logs["step_count"] = tensordict_data["step_count"].max().item()
            stepcount_str = f"step count (max): {logs['step_count']}"
            logs["lr"] = optim.param_groups[0]["lr"]
            lr_str = f"lr policy: {logs['lr']: 4.6f}"
            if args.track_bellman_violation:
                bm_viol = calculate_bellman_violation(10, 
                                                    value_net,
                                                    state_space, 
                                                    policy_module,
                                                    base_env, 
                                                    gamma,
                                                    after_batch_transform=after_batch_transform)
                                            
                                                                
                logs["bellman_violation_mean"] = bm_viol.flatten().mean().item()
                logs["bellman_violation_max"] = bm_viol.flatten().max().item()
                logs["bellman_violation_std"] = bm_viol.flatten().std().item()
            if i % 10 == 0 and args.eval:
                eval_logs, eval_str = evaluate_policy(env, policy_module, max_rollout_len)
                for key, val in eval_logs.items():
                    logs[key] = val
            if args.track:
                wandb.log({**logs})
            else:
                pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
            # scheduler.step()
        wandb.finish() 
        collector.shutdown()
    #######################
    # Evaluation:
    #######################
    if args.save:
        print("Saving")
        # Save the model
        torch.save(policy_module.state_dict(), "models/weights/ppo_policy_safe_integrator" \
            + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pth")
        torch.save(value_module.state_dict(), "models/weights/ppo_value_safe_integrator" \
            + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pth")
    if args.eval:
        print("Evaluation")
        eval_logs, eval_str = evaluate_policy(env, policy_module, max_rollout_len)
        print(eval_str)
    if args.plot_value and args.plot_traj > 0:
        print("Plotting value function")
        value_function_resolution = 10
        value_landscape = plot_value_function_integrator(max_x1, 
                                       max_x2,
                                       value_function_resolution,
                                       value_net)
    else:
        value_landscape = None
    if args.plot_traj > 0:
        plot_integrator_trajectories(env, 
                                    policy_module,
                                    max_rollout_len,
                                    args.plot_traj,
                                    value_net)
        print("Plotted trajectories")
    if args.plot_bellman_violation:

        print("Calculating and plotting Bellman violation")
        obs_norm_loc = env.transform[3].loc
        obs_norm_scale = env.transform[3].scale
        bm_viol = calculate_bellman_violation(10, 
                                            value_net,
                                            state_space, 
                                            policy_module,
                                            base_env,
                                            gamma,
                                            after_batch_transform=after_batch_transform)
        plt.figure(figsize=(10, 10))
        # Better with contourf, or imshow or maybe surface plot or pcolormesh
        plt.contourf(bm_viol,cmap="coolwarm")
        plt.colorbar()
        plt.title("Bellman violation")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.savefig("results/ppo_safe_integrator_bellman_violation" +\
            datetime.now().strftime("%Y%m%d-%H%M%S") + ".pdf")
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
        plt.savefig("results/ppo_safe_integrator_statistics.png")
