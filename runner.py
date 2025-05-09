from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import wandb
from typing import List, Dict, Any
from tensordict.nn import TensorDictModule
from torch import nn
from tensordict.nn.distributions import NormalParamExtractor
from tensordict import TensorDict
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
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
)
from torch import multiprocessing
from torchrl.envs.utils import ExplorationType
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives.value import GAE
from envs.integrator import MultiObjectiveDoubleIntegratorEnv, plot_integrator_trajectories, plot_value_function_integrator
from datetime import datetime
import argparse
from results.evaluate import PolicyEvaluator, calculate_bellman_violation
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
    parser.add_argument("--max_rollout_len", type=int, default=32, help="Maximum rollout length")
    parser.add_argument("--track", action="store_true", default=False, help="Track the training with wandb")
    parser.add_argument("--wandb_project", type=str, default="ppo_safe_integrator", help="Wandb project name")
    parser.add_argument("--experiment_name", type=str, default=None, help="Wandb experiment name")
    parser.add_argument("--track_bellman_violation", action="store_true", 
                        default=False, help="Track the Bellman violation")
    parser.add_argument("--plot_bellman_violation", action="store_true",
                        default=False, help="Plot the Bellman violation of trained value function and policy")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return vars(parser.parse_args())
    
if __name__ == "__main__":
    args = parse_args() 

    # Set seed
    torch.manual_seed(args["seed"])
        
    #######################
    # Parallelization:
    #######################
    multiprocessing.set_start_method("spawn", force=True)
    is_fork = multiprocessing.get_start_method(allow_none=True) == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    parameters = TensorDict({
            "dt": 0.05,
            "max_x1": 1.0,
            "max_x2": 1.0,
            "max_input": 1.0,
            "reference_amplitude": 1.1,
            "reference_frequency": 0.1, # rule of thumb: < max_u/(2*pi*sqrt(A))
        },[],device=device)
        
    state_space = {"x1": {"low": -parameters["max_x1"], "high": parameters["max_x1"]},
                    "x2": {"low": -parameters["max_x2"], "high": parameters["max_x2"]},}
    #######################
    # Arguments:
    #######################
    args["gamma"] = 0.95
    args["num_epochs"] = 10 # optimization steps per batch of data collected
    args["frames_per_batch"] = int(2**12)
    args["sub_batch_size"] = int(2**8)
    args["max_grad_norm"] = 1.0
    args["total_frames"] = int(2**19)
    args["device"] = device
    args["clip_epsilon"] = 0.2
    args["lmbda1"] = 0.1
    args["lmbda2"] = 0.95
    args["critic_coef"] = 1.0
    args["supervision_coef"] = 1.0
    args["collision_buffer_size"] = args["frames_per_batch"]
    args["loss_critic_type"] = "smooth_l1"
    args["optim_kwargs"] = {"lr": 5e-5}
    args["bellman_eval_res"] = 10 #Resolution of grid over state space used for calculating Bellman violation
    args["state_space"] = state_space
    args["primary_reward_key"] = "r1"
    args["secondary_reward_key"] = "r2"
    args["entropy_coef"] = 0.001
    args["num_parallel_env"] = int(args["frames_per_batch"] / (args["max_rollout_len"]))

    #######################
    # Environment:
    #######################
    base_env = MultiObjectiveDoubleIntegratorEnv(batch_size=args.get("num_parallel_env"),
                                                 device=device,
                                                 td_params=parameters,
                                                 seed=args["seed"])
    obs_signals = ["x1","x2"]
    ref_signals = ["y1_ref","y2_ref"]
    transforms = [
            UnsqueezeTransform(in_keys=obs_signals+ref_signals, 
                               dim=-1,
                               in_keys_inv=obs_signals+ref_signals,),
            CatTensors(in_keys=obs_signals, out_key= "obs",del_keys=False,dim=-1),
            CatTensors(in_keys=ref_signals, out_key= "ref",del_keys=False,dim=-1),
            ObservationNorm(in_keys=["obs"], out_keys=["obs"]),
            ObservationNorm(in_keys=["ref"], out_keys=["ref"]),
            CatTensors(in_keys=["obs","ref"], out_key="obs_extended",del_keys=False,dim=-1),
            DoubleToFloat(),
            StepCounter(max_steps=args["max_rollout_len"])]
    env = TransformedEnv(
        base_env,
        Compose(
            *transforms
        )
    ).to(device)
    env.transform[3].init_stats(num_iter=1000,reduce_dim=(0,1),cat_dim=1)
    env.transform[4].init_stats(num_iter=1000,reduce_dim=(0,1),cat_dim=1)
    
    #######################
    # Models:
    #######################


    nn_net_config = {
        "name": "feedforward",
        "eps": 1e-2,
        "layers": [64, 64],
        "activation": nn.ReLU(),
        "device": device,
        "input_size": len(obs_signals),
        "bounded": True,
    }
    actor_net = nn.Sequential()
    layers = [len(ref_signals+obs_signals)] + nn_net_config["layers"] 
    for i in range(len(layers)-1):
        actor_net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i + 1],device=device))
        actor_net.add_module(f"activation_{i}", nn_net_config["activation"])
    actor_net.add_module("output", nn.Linear(layers[-1], 2*env.action_spec.shape[-1],device=device))
    actor_net.add_module("param_extractor", NormalParamExtractor())


    policy_module = ProbabilisticActor(
        module=TensorDictModule(actor_net,
                                in_keys=["obs_extended"],
                                out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"],
        spec=env.action_spec,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": -parameters["max_input"],
            "high": parameters["max_input"],
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
    layers = [len(ref_signals+obs_signals)] + nn_net_config["layers"]
    for i in range(len(layers)-1):
        value_net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i + 1],device=device))
        value_net.add_module(f"activation_{i}", nn_net_config["activation"])
    value_net.add_module("output", nn.Linear(layers[-1], 1,device=device))
    value_module = ValueOperator(
        module=value_net,
        in_keys=["obs_extended"],
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
                                rollout_len=args["max_rollout_len"],
                                keys_to_log=[args.get("primary_reward_key"),
                                             args.get("secondary_reward_key"),
                                             "step_count"])
    if args.get("track_bellman_violation",False):
        base_env = env.base_env if isinstance(env, TransformedEnv) else env
        def eval_func(data):
            logs = defaultdict(list)
            bm_viol, *_ = calculate_bellman_violation(
                args.get("bellman_eval_res",10),
                CBF_module,
                state_space, 
                policy_module,
                MultiObjectiveDoubleIntegratorEnv, 
                args.get("gamma"),
                transforms=env.transform[:-1]
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
        collector = SyncDataCollector(
            create_env_fn=env,
            policy=policy_module,
            frames_per_batch=args["frames_per_batch"],
            total_frames=args["total_frames"],
            split_trajs=False,
            device=device,
            exploration_type=ExplorationType.RANDOM)

        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=args["frames_per_batch"]),
            sampler=SamplerWithoutReplacement(),
        )
        optim = torch.optim.Adam
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optim, args["total_frames"] // args["frames_per_batch"], 1e-8
        # )
        ppo_entity.train(
            policy_module=policy_module,
            V_primary=CBF_module,
            V_secondary=value_module,
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
        plot_value_function_integrator(parameters["max_x1"], 
                                       parameters["max_x2"],
                                       resolution,
                                       CBF_module,
                                       transforms=env.transform[:-1])
    if args.get("plot_traj") > 0:
        plot_integrator_trajectories(env, 
                                    policy_module,
                                    args["max_rollout_len"],
                                    args.get("plot_traj"),
                                    CBF_module)
        print("Plotted trajectories")
    if args.get("plot_bellman_violation"):
        print("Calculating and plotting Bellman violation")
        obs_norm_loc = env.transform[3].loc
        obs_norm_scale = env.transform[3].scale
        bm_viol,mesh = calculate_bellman_violation(10, 
                                            CBF_module,
                                            state_space, 
                                            policy_module,
                                            MultiObjectiveDoubleIntegratorEnv,
                                            args.get("gamma"),
                                            transforms=env.transform[:-1])
        X = mesh[0].reshape(bm_viol.shape)
        Y = mesh[1].reshape(bm_viol.shape)
        plt.figure(figsize=(10, 10))
        # Better with contourf, or imshow or maybe surface plot or pcolormesh
        plt.contourf(X,Y,bm_viol,cmap="coolwarm")
        plt.colorbar()
        plt.title("Bellman violation")
        plt.xlabel("x1")
        plt.ylabel("x2")
        if wandb.run is not None:
            wandb.log({"bellman_violation": wandb.Image(plt)})
        else:
            plt.savefig("results/ppo_safe_integrator_bellman_violation" +\
                datetime.now().strftime("%Y%m%d-%H%M%S") + ".pdf")
    if wandb.run is not None:
        wandb.finish()