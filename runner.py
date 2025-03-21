from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from typing import List, Dict, Any
from tensordict.nn import TensorDictModule
from torch import nn
from tensordict.nn.distributions import NormalParamExtractor
from tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.collectors import MultiSyncDataCollector
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    CatTensors,
    UnsqueezeTransform,
    EnvCreator,
    BatchSizeTransform
)
from torch import multiprocessing
from torchrl.envs.utils import ExplorationType
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives.value import GAE
from envs.integrator import SafeDoubleIntegratorEnv, plot_integrator_trajectories, plot_value_function_integrator
from datetime import datetime
import argparse
from results.evaluate import evaluate_policy, calculate_bellman_violation
from utils.utils import reset_batched_env   
import wandb
from models.factory import SafetyValueFunctionFactory
from algorithms.ppo import PPO


multiprocessing.set_start_method("spawn", force=True)
is_fork = multiprocessing.get_start_method(allow_none=True) == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)


def parse_args()->Dict[str,Any]:
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
    parser.add_argument("--max_rollout_len", type=int, default=100, help="Maximum rollout length")
    parser.add_argument("--track", action="store_true", default=False, help="Track the training with wandb")
    parser.add_argument("--wandb_project", type=str, default="ppo_safe_integrator", help="Wandb project name")
    parser.add_argument("--experiment_name", type=str, default=None, help="Wandb experiment name")
    parser.add_argument("--track_bellman_violation", action="store_true", 
                        default=False, help="Track the Bellman violation")
    parser.add_argument("--plot_bellman_violation", action="store_true",
                        default=False, help="Plot the Bellman violation of trained value function and policy")
    
    return vars(parser.parse_args())
    
if __name__ == "__main__":
    args = parse_args() 

    #######################
    # Hyperparameters:
    #######################
    num_cells = 64
    max_input = 1.0
    max_x1 = 1.0
    max_x2 = 1.0
    dt = 0.05
    frames_per_batch = int(2**17)
    lr = 1e-5
    max_grad_norm = 1.0
    total_frames = int(2**21)
    num_epochs = 20  # optimization steps per batch of data collected
    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    critic_coef = 1.0
    loss_critic_type = "smooth_l1"
    sub_batch_size = int(2**10)
    lmbda = 0.95
    entropy_eps = 0.0
    value_net_config = {
        "name": "feedforward",
        "eps": 1e-2,
        "layers": [64, 64],
        "activation": nn.ReLU(),
        "device": device,
        "input_size": 2,
        "bounded": False,
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

    multiprocessing.set_start_method("spawn", force=True)
    is_fork = multiprocessing.get_start_method(allow_none=True) == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    #######################
    # Environment:
    #######################
    state_space = {"x1": {"low": -max_x1, "high": max_x1},
                    "x2": {"low": -max_x2, "high": max_x2}}

    max_rollout_len = args.get("max_rollout_len")
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
    #######################
    # Arguments:
    #######################
    args["gamma"] = gamma
    args["num_epochs"] = num_epochs
    args["frames_per_batch"] = frames_per_batch
    args["sub_batch_size"] = sub_batch_size
    args["max_grad_norm"] = max_grad_norm
    args["total_frames"] = total_frames
    args["entropy_eps"] = entropy_eps
    args["device"] = device
    args["clip_epsilon"] = clip_epsilon
    args["lmbda"] = lmbda
    args["critic_coef"] = critic_coef
    args["loss_critic_type"] = loss_critic_type
    args["optim_kwargs"] = {"lr": lr}
    args["bellman_eval_res"] = 10
    args["state_space"] = state_space
    
    
    #######################
    # Models:
    #######################

    # Handle both batch-locked and unbatched action specs
    action_high = (env.action_spec_unbatched.high if hasattr(env, "action_spec_unbatched") 
                                                    else env.action_spec.high)
    action_low = (env.action_spec_unbatched.low if hasattr(env, "action_spec_unbatched") 
                                                    else env.action_spec.low)
    observation_size_unbatched = (env.obs_size_unbatched if hasattr(env, "obs_size_unbatched") 
                                                            else env.observation_space.shape[0])
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
    if args.get("load_policy") is not None:
        policy_module.load_state_dict(torch.load(args.get("load_policy")))
        print("Policy loaded")

    value_net = SafetyValueFunctionFactory.create(**value_net_config)
    value_module = ValueOperator(
        module=value_net,
        in_keys=["obs"],
    )
    if args.get("load_value") is not None:
        value_module.load_state_dict(torch.load(args.get("load_value")))
        print("Value function loaded") 


    #######################
    # Training:
    #######################
    ppo_entity = PPO()
    ppo_entity.setup(args)

    if args.get("track_bellman_violation",False):
        base_env = env.base_env if isinstance(env, TransformedEnv) else env
        value_net = value_module.module
        def eval_func(data):
            logs = defaultdict(list)
            bm_viol = calculate_bellman_violation(
                args.get("bellman_eval_res",10),
                value_net,
                state_space, 
                policy_module,
                base_env, 
                gamma,
                after_batch_transform=after_batch_transform
            )
            eval_logs= evaluate_policy(env, policy_module, max_rollout_len)
            logs.update(eval_logs)
            logs["bellman_violation_mean"] = bm_viol.flatten().mean().item()
            logs["bellman_violation_max"] = bm_viol.flatten().max().item()
            logs["bellman_violation_std"] = bm_viol.flatten().std().item()
            return logs
    else:
        eval_func = lambda x: evaluate_policy(env, policy_module, max_rollout_len)
    if args.get("train"):
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
        optim = torch.optim.Adam
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optim, total_frames // frames_per_batch, 1e-8
        # )
        ppo_entity.train(
            policy_module=policy_module,
            value_module=value_module,
            optim=optim,
            collector=collector,
            replay_buffer=replay_buffer,
            eval_func=eval_func 
        ) 
        
    if args.get("save"):
        print("Saving")
        # Save the model
        torch.save(policy_module.state_dict(), "models/weights/ppo_policy_safe_integrator" \
            + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pth")
        torch.save(value_module.state_dict(), "models/weights/ppo_value_safe_integrator" \
            + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pth")
    #######################
    # Evaluation:
    #######################
    if args.get("eval"):
        print("Evaluation")
        eval_logs, eval_str = evaluate_policy(env, policy_module, max_rollout_len)
        print(eval_str)
    if args.get("plot_value") and args.get("plot_traj") > 0:
        print("Plotting value function")
        value_function_resolution = 10
        plot_value_function_integrator(max_x1, 
                                       max_x2,
                                       value_function_resolution,
                                       value_net)
    if args.get("plot_traj") > 0:
        plot_integrator_trajectories(env, 
                                    policy_module,
                                    max_rollout_len,
                                    args.get("plot_traj"),
                                    value_net)
        print("Plotted trajectories")
    if args.get("plot_bellman_violation"):
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