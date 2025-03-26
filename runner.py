from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from typing import List, Dict, Any
from tensordict.nn import TensorDictModule
from torch import nn
from tensordict.nn.distributions import NormalParamExtractor
from tensordict import TensorDict
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.collectors import MultiSyncDataCollector
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement, PrioritizedSampler
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
from envs.integrator import MultiObjectiveDoubleIntegratorEnv, plot_integrator_trajectories, plot_value_function_integrator
from datetime import datetime
import argparse
from results.evaluate import PolicyEvaluator, calculate_bellman_violation
from utils.utils import reset_batched_env   
import wandb
from models.factory import SafetyValueFunctionFactory
from algorithms.hippo import HierarchicalPPO as HiPPO
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
    parser.add_argument("--load_CBF", type=str, default=None, help="Path to load CBF")
    parser.add_argument("--train", action="store_true", default=False, help="Train the model")
    parser.add_argument("--eval", action="store_true", default=False, help="Evaluate the model, both during and after training") 
    parser.add_argument("--plot_traj", type=int, default=0, help="Number of trajectories to plot")
    parser.add_argument("--save", action="store_true", default=False, help="Save the models")
    parser.add_argument("--plot_CBF", action="store_true", default=False, help="Plot the value function landscape")
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
    frames_per_batch = int(2**12)
    lr = 5e-5
    max_grad_norm = 1.0
    total_frames = int(2**20)
    num_epochs = 10  # optimization steps per batch of data collected
    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    critic_coef = 1.0
    loss_critic_type = "smooth_l1"
    sub_batch_size = int(2**8)
    lmbda = 0.95
    entropy_eps = 0.0
    nn_net_config = {
        "name": "feedforward",
        "eps": 1e-2,
        "layers": [64, 64],
        "activation": nn.ReLU(),
        "device": device,
        "input_size": 2,
        "bounded": True,
    }
        
    #######################
    # Parallelization:
    #######################
    max_rollout_len = args.get("max_rollout_len")
    if device.type == "cuda":
        batches_per_process = int(frames_per_batch / (max_rollout_len))
        num_workers = 1
    else:
        batches_per_process = int(frames_per_batch / (max_rollout_len))
        num_workers = 1

    multiprocessing.set_start_method("spawn", force=True)
    is_fork = multiprocessing.get_start_method(allow_none=True) == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    state_space = {"x1": {"low": -max_x1, "high": max_x1},
                    "x2": {"low": -max_x2, "high": max_x2}}

    parameters = TensorDict({
        "params" : TensorDict({
            "dt": dt,
            "max_x1": max_x1,
            "max_x2": max_x2,
            "max_input": max_input,
        },[],
        device=device)
        },[],device=device)
        
    #######################
    # Arguments:
    #######################
    args["gamma"] = 0.97
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
    # P(i) = p_i^alpha / sum(p_i^alpha)
    # w(i) = 1/(N*P(i))^beta
    args["alpha"] = 0.0
    args["beta"] = 1.0
    args["primary_reward_key"] = "r1"
    args["secondary_reward_key"] = "r2"

    #######################
    # Environment:
    #######################
    base_env = MultiObjectiveDoubleIntegratorEnv(device=device,
                                                 td_params=parameters)
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


    actor_net = nn.Sequential()
    layers = [observation_size_unbatched] + nn_net_config["layers"] 
    for i in range(len(layers)-1):
        actor_net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i + 1],device=device))
        actor_net.add_module(f"activation_{i}", nn_net_config["activation"])
    actor_net.add_module("output", nn.Linear(layers[-1], 2*env.action_spec.shape[-1],device=device))
    actor_net.add_module("param_extractor", NormalParamExtractor())


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

    CBF_net = SafetyValueFunctionFactory.create(**nn_net_config)
    CBF_module = ValueOperator(
        module=CBF_net,
        in_keys=["obs"],
        out_keys=["V1"],
    )
    if args.get("load_CBF") is not None:
        CBF_module.load_state_dict(torch.load(args.get("load_CBF")))
        print("CBF network loaded") 

    value_net = nn.Sequential()
    layers = [observation_size_unbatched] + nn_net_config["layers"]
    for i in range(len(layers)-1):
        value_net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i + 1],device=device))
        value_net.add_module(f"activation_{i}", nn_net_config["activation"])
    value_net.add_module("output", nn.Linear(layers[-1], 1,device=device))
    value_module = ValueOperator(
        module=value_net,
        in_keys=["obs"],
        out_keys=["V2"]
    )
    if args.get("load_value") is not None:
        value_module.load_state_dict(torch.load(args.get("load_value")))
        print("Value network loaded")
        

    #######################
    # Training:
    #######################
    ppo_entity = HiPPO()
    ppo_entity.setup(args)
    evaluator = PolicyEvaluator(env=env,
                                policy_module=policy_module,
                                rollout_len=max_rollout_len,
                                keys_to_log=[args.get("primary_reward_key"),
                                             args.get("secondary_reward_key"),
                                             "step_count"])
    if args.get("track_bellman_violation",False):
        base_env = env.base_env if isinstance(env, TransformedEnv) else env
        def eval_func(data):
            logs = defaultdict(list)
            bm_viol = calculate_bellman_violation(
                args.get("bellman_eval_res",10),
                CBF_net,
                state_space, 
                policy_module,
                base_env, 
                args.get("gamma"),
                after_batch_transform=after_batch_transform
            )
            eval_logs = evaluator.evaluate_policy()
            logs.update(eval_logs)
            logs["bellman_violation_mean"] = bm_viol.flatten().mean().item()
            logs["bellman_violation_max"] = bm_viol.flatten().max().item()
            logs["bellman_violation_std"] = bm_viol.flatten().std().item()
            return logs
    else:
        eval_func = lambda x: evaluator.evaluate_policy()
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

        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=frames_per_batch),
            sampler=PrioritizedSampler(max_capacity=frames_per_batch,
                                        alpha=args.get("alpha",0.6),
                                        beta=args.get("beta",0.4)),
        )
        # replay_buffer = TensorDictReplayBuffer(
        #     storage=LazyTensorStorage(max_size=frames_per_batch),
        #     sampler=SamplerWithoutReplacement(),
        # )
        optim = torch.optim.Adam
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optim, total_frames // frames_per_batch, 1e-8
        # )
        ppo_entity.train(
            policy_module=policy_module,
            V_primary = CBF_module,
            V_secondary = value_module,
            optim=optim,
            collector=collector,
            replay_buffer=replay_buffer,
            eval_func=eval_func 
        ) 
        
    if args.get("save"):
        print("Saving")
        # Save the models
        torch.save(policy_module.state_dict(), "models/weights/ppo_policy_safe_integrator" \
            + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pth")
        torch.save(value_module.state_dict(), "models/weights/ppo_value_safe_integrator" \
            + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pth")
        torch.save(CBF_module.state_dict(), "models/weights/ppo_CBF_safe_integrator" \
            + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pth")

    #######################
    # Evaluation:
    #######################
    if args.get("eval"):
        print("Evaluation")
        eval_logs = evaluator.evaluate_policy()
        eval_str = ",".join([f"{key}: {val}" for key,val in eval_logs.items()])
        print(eval_str)
    if args.get("plot_CBF") and args.get("plot_traj") > 0:
        print("Plotting CBF")
        resolution = 10
        plot_value_function_integrator(max_x1, 
                                       max_x2,
                                       resolution,
                                       CBF_net)
    if args.get("plot_traj") > 0:
        plot_integrator_trajectories(env, 
                                    policy_module,
                                    max_rollout_len,
                                    args.get("plot_traj"),
                                    CBF_net)
        print("Plotted trajectories")
    if args.get("plot_bellman_violation"):
        print("Calculating and plotting Bellman violation")
        obs_norm_loc = env.transform[3].loc
        obs_norm_scale = env.transform[3].scale
        bm_viol = calculate_bellman_violation(10, 
                                            CBF_net,
                                            state_space, 
                                            policy_module,
                                            base_env,
                                            args.get("gamma"),
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