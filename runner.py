import torch
import wandb
from torch import multiprocessing
import argparse
from typing import Dict, Any
from runners.double_integrator import DoubleIntegratorRunner
from datetime import datetime


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
    parser.add_argument("--load_cdf", type=str, default=None, help="Path to load CBF")
    parser.add_argument("--train", action="store_true", default=False, help="Train the model")
    parser.add_argument("--eval", action="store_true", default=False, help="Evaluate the model (after training)") 
    parser.add_argument("--save", action="store_true", default=False, help="Save the models")
    parser.add_argument("--track", action="store_true", default=False, help="Track the training with wandb")
    parser.add_argument("--wandb_project", type=str, default="ppo_safe_integrator", help="Wandb project name")
    parser.add_argument("--experiment_name", type=str, default=None, help="Wandb experiment name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--plot", action="store_true", default=False, help="Plot the results")
    return vars(parser.parse_args())
    
if __name__ == "__main__":
    args = parse_args() 

    # Set seed
        
    args["seed"] = 0
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

    #######################
    # Arguments:
    #######################

    args["env_params"] = {
        "dt": 0.05,
        "max_x1": 1.0,
        "max_x2": 1.0,
        "max_input": 1.0,
        "reference_amplitude": 1.1,
        "reference_frequency": 0.1, # rule of thumb: < max_u/(2*pi*sqrt(A))
    }
    args["gamma"] = 0.95
    args["plot_traj"] = 32
    args["max_rollout_len"] = 32
    args["plot_bellman_violation"] = True 
    args["plot_cdf"] = True
    args["track_bellman_violation"] = True
    
    args["num_epochs"] = 10 # optimization steps per batch of data collected
    args["frames_per_batch"] = int(2**12)
    args["sub_batch_size"] = int(2**8)
    args["max_grad_norm"] = 1.0
    args["total_frames"] = int(2**18)
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
    args["state_space"] = {
        "x1": {"low": -args["env_params"]["max_x1"],
                "high": args["env_params"]["max_x1"]},
        "x2": {"low": -args["env_params"]["max_x2"],
                "high": args["env_params"]["max_x2"]}
    }
    args["primary_reward_key"] = "r1"
    args["secondary_reward_key"] = "r2"
    args["entropy_coef"] = 0.001
    args["num_parallel_env"] = int(args["frames_per_batch"] / (args["max_rollout_len"]))

    runner = DoubleIntegratorRunner(device=device)
    runner.setup(args)
    if args.get("cdf_path") is not None:
        runner.load(cdf_path=args["cdf_path"])
    if args.get("policy_path") is not None:
        runner.load(policy_path=args["policy_path"])
    if args.get("value_path") is not None:
        runner.load(value_path=args["value_path"])
    if args.get("train", False): 
        runner.train()
    if args.get("save", False):
        cdf_path = "double_integrator_cdf" + datetime.strftime("%Y%m%d-%H%M%S") + ".pt"
        policy_path = "double_integrator_policy" + datetime.strftime("%Y%m%d-%H%M%S") + ".pt"
        value_path = "double_integrator_value" + datetime.strftime("%Y%m%d-%H%M%S") + ".pt"
        runner.save(cdf_path=cdf_path,
                    policy_path=policy_path,
                    value_path=value_path) 
    if args.get("eval", False):
        logs = runner.evaluate()
        print("Evaluation logs: ", logs)
    if args.get("plot", False):
        runner.plot_results()
    
    if wandb.run is not None:
        wandb.finish()