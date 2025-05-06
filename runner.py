from algorithms.hippo import HierarchicalPPO as HiPPO
from typing import Dict, Any
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.utils import ExplorationType
from typing import Optional
from results.evaluate import PolicyEvaluator, calculate_bellman_violation
import torch
from torch import nn, multiprocessing
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    CatTensors,
    UnsqueezeTransform,
)
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from utils.utils import get_config_value
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import wandb
import argparse
import yaml
from envs.integrator import(
    plot_integrator_trajectories,
    plot_value_function_integrator
)
from envs import make_env
from models.factory import PolicyFactory, ValueFactory
from torchrl.envs import GymEnv
import safety_gym
from torchrl.envs.utils import step_mdp
import time

ACTIVATION_MAP = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "gelu": nn.GELU(),
}
class Runner():
    def __init__(self, device):
        self.device = device
        self.env = None
        self.policy_module = None
        self.cdf_module = None
        self.value_module = None
        self.args = None
    def train(self):
        if self.args == None:
            raise ValueError("Setup the runner before training")
        ppo_entity = HiPPO()
        HiPPO_args = self.args["algorithm"]
        HiPPO_args["device"] = self.device
        HiPPO_args["gamma"] = self.args["env"]["gamma"]
        HiPPO_args["safety_obs_key"] = self.args["env"]["cfg"]["safety_obs_key"]
        ppo_entity.setup(HiPPO_args)
        print("Training...")

        if self.args.get("train",False):
            collector = SyncDataCollector(
                create_env_fn=self.env,
                policy=self.policy_module,
                frames_per_batch=HiPPO_args["frames_per_batch"],
                total_frames=HiPPO_args["total_frames"],
                split_trajs=False,
                device=self.device,
                exploration_type=ExplorationType.RANDOM,
                reset_at_each_iter=True)

            replay_buffer = TensorDictReplayBuffer(
                storage=LazyTensorStorage(max_size=HiPPO_args["frames_per_batch"]),
                sampler=SamplerWithoutReplacement(),
            )
            optim = torch.optim.Adam
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optim, args["total_frames"] // args["frames_per_batch"], 1e-8
            # )
            ppo_entity.train(
                policy_module=self.policy_module,
                V_primary=self.cdf_module,
                V_secondary=self.value_module,
                optim=optim,
                collector=collector,
                replay_buffer=replay_buffer,
                eval_func=self.evaluate
            ) 

    def setup(self,args):
        self.args = args
        warn_str = "Warning: max_input not set in env params. Defaulting to infinity"
        default_params = {
            "max_input": float("inf"),
        }
        self.params = get_config_value(args["env"]["cfg"], "params", default_params, warn_str) 

        self.env = make_env(args["env"]["name"], args["env"]["cfg"], self.device)
        
        #######################
        # Models:
        #######################


        full_obs_key = args["env"]["cfg"]["full_obs_key"]
        safety_obs_key = args["env"]["cfg"]["safety_obs_key"] 
        full_obs_size = self.env.observation_spec[full_obs_key].shape[-1]
        safety_obs_size = self.env.observation_spec[safety_obs_key].shape[-1]
        cdf_net_config = args["models"]["cdf_net"]
        cdf_net_config = {
            "name": cdf_net_config["name"],
            "eps": cdf_net_config.get("eps", 0.0),
            "layers": cdf_net_config["layers"],
            "activation": ACTIVATION_MAP[cdf_net_config["activation"]],
            "device": self.device,
            "input_size": safety_obs_size, #Operating under the assumption that the cdd
            # function only depends on the state, not on any reference signals
        }
        value_net_config = args["models"]["value_net"]
        value_net_config = {
            "name": value_net_config["name"],
            "eps": value_net_config.get("eps", 0.0),
            "layers": value_net_config["layers"],
            "activation": ACTIVATION_MAP[value_net_config["activation"]],
            "device": self.device,
            "input_size": full_obs_size,
        } 
        policy_net_config = args["models"]["policy_net"]
        policy_net_config = {
            "name": policy_net_config["name"],
            "layers": policy_net_config["layers"],
            "activation": ACTIVATION_MAP[policy_net_config["activation"]],
            "device": self.device,
            "input_size": full_obs_size,
            "action_size": self.env.action_spec.shape[-1],
        }
            
        actor_net = PolicyFactory.create(**policy_net_config)
        self.policy_module = ProbabilisticActor(
            module=TensorDictModule(actor_net,
                                    in_keys=[full_obs_key],
                                    out_keys=["loc", "scale"]),
            in_keys=["loc", "scale"],
            spec=self.env.action_spec,
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": -self.params["max_input"],
                "high": self.params["max_input"],
            },
            return_log_prob=True,
        )

        value_net = ValueFactory.create(**value_net_config)
        self.value_module = ValueOperator(
            module=value_net,
            in_keys=[full_obs_key],
            out_keys=["V2"]
        )

        cdf_net = ValueFactory.create(**cdf_net_config)
        self.cdf_module = ValueOperator(
            module=cdf_net,
            in_keys=[safety_obs_key],
            out_keys=["V1"],
        )

        self.evaluator = PolicyEvaluator(env=self.env,
                                    policy_module=self.policy_module,
                                    rollout_len=self.args["env"]["cfg"]["max_steps"],
                                    keys_to_log=[self.args["algorithm"]["primary_reward_key"],
                                                self.args["algorithm"]["secondary_reward_key"],
                                                "step_count"])
        
    def visualize_results(self):
        """
        Visualize the results of the training.
        This takes different forms depending on the environment.
        """
        if self.args == None:
            raise ValueError("Setup the runner before plotting")
        if self.args["env"]["name"] == "double_integrator": 
            self.plot_integrator_results()
        elif self.args["env"]["name"].startswith("Safexp"):
            self.render_safety_gym_results()
        else:
            raise ValueError("No visualization method for this environment")
    def evaluate(self):
        if self.args == None:
            raise ValueError("Setup the runner before evaluating")
        eval_args = self.args.get("evaluation", {})
        logs = defaultdict(list)
        eval_logs = self.evaluator.evaluate_policy()
        logs.update(eval_logs)
        if eval_args.get("track_bellman_violation",False):
            bm_viol, *_ = calculate_bellman_violation(
                eval_args.get("bellman_eval_res",10),
                self.cdf_module,
                eval_args["state_space"],
                self.policy_module,
                self.env_maker, 
                self.args["env"]["gamma"],
                transforms=self.env.transform[:-1]
            )
            logs["bellman_violation_mean"] = bm_viol.flatten().mean().item()
            logs["bellman_violation_max"] = bm_viol.flatten().max().item()
            logs["bellman_violation_std"] = bm_viol.flatten().std().item()
        return logs
    def save(self,
             cdf_path:Optional[str]=None, 
             policy_path:Optional[str]=None, 
             value_path:Optional[str]=None):
        if self.args is None:
            raise ValueError("Setup the runner before saving")
        if cdf_path is not None:
            torch.save(self.cdf_module.state_dict(), cdf_path)
        if policy_path is not None:
            torch.save(self.policy_module.state_dict(),policy_path)
        if value_path is not None:
            torch.save(self.value_module.state_dict(), value_path)
        print("Models saved")
    def load(self, 
             cdf_path:Optional[str]=None, 
             policy_path:Optional[str]=None, 
             value_path:Optional[str]=None):
        if self.args is None:
            raise ValueError("Setup the runner before loading")
        if value_path is not None:
            self.value_module.load_state_dict(torch.load(value_path))
            print("Value network loaded")       
        if policy_path is not None:
            self.policy_module.load_state_dict(torch.load(policy_path))
            print("Policy loaded")
        if cdf_path is not None:
            self.cdf_module.load_state_dict(torch.load(cdf_path))
            print("CDF network loaded")
    # Bellman violation uses a custom env with a different batch size
    def env_maker(self, batch_size:int, device:Optional[torch.device]=None):
        env_cfg = self.args["env"]["cfg"]
        env_cfg["num_parallel_env"] = batch_size
        env = make_env(self.args["env"]["name"], env_cfg, device=device)
        return env.base_env
    def plot_integrator_results(self):
        plotting_args = self.args.get("plot", {})
        plotting_args["max_steps"] = self.args["env"]["cfg"]["max_steps"]
        if plotting_args.get("cdf",False):
            print("Plotting cdf")
            resolution = 10
            plot_value_function_integrator(self.params["max_x1"], 
                                        self.params["max_x2"],
                                        resolution,
                                        self.cdf_module,
                                        transforms=self.env.transform[:-1])
        if plotting_args.get("num_trajs",0) > 0:
            plot_integrator_trajectories(self.env, 
                                        self.policy_module,
                                        plotting_args["max_steps"],
                                        plotting_args["num_trajs"],
                                        self.cdf_module)
            print("Plotted trajectories")
        if plotting_args.get("bellman_violation",False):
            print("Calculating and plotting Bellman violation")
            bm_viol,mesh = calculate_bellman_violation(10, 
                                                self.cdf_module,
                                                plotting_args["state_space"], 
                                                self.policy_module,
                                                self.env_maker,
                                                self.args["env"]["gamma"],
                                                transforms=self.env.transform[:-1])
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
            
    def render_safety_gym_results(self):
        render_args = self.args.get("render", {})
        if render_args.get("render", False):
        # rendering only works with unbatched GymEnvs
            env = GymEnv(self.args["env"]["name"],device=self.device)
            env.seed(self.args["env"]["cfg"]["seed"])
            td = env.reset()
            dt = 1.0/render_args.get("fps",60)
            for _ in range(render_args["num_frames"]):
                time.sleep(dt)
                env.render()
                with torch.no_grad():
                    td = self.policy_module(td)
                td = env.step(td)
                td = step_mdp(td)
                done = td["done"].any()
                if done:
                    td = env.reset()
            env.close()
                

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
    parser.add_argument("--load_cdf", type=str, default=None, help="Path to load CDF")
    parser.add_argument("--train", action="store_true", default=False, help="Train the model")
    parser.add_argument("--eval", action="store_true", default=False, help="Evaluate the model (after training)") 
    parser.add_argument("--save", action="store_true", default=False, help="Save the models")
    parser.add_argument("--track", action="store_true", default=False, help="Track the training with wandb")
    parser.add_argument("--wandb_project", type=str, default="hippo", help="Wandb project name")
    parser.add_argument("--experiment_name", type=str, default=None, help="Wandb experiment name")
    parser.add_argument("--visualize", action="store_true", default=False, help="Visualize the results")
    parser.add_argument("--config", type=str, default="configs/hippo_double_integrator.yaml", help="Path to config file(yaml)")
    return vars(parser.parse_args())
    
if __name__ == "__main__":
    args = parse_args() 
    with open(args["config"], "r") as f:
        args.update(yaml.safe_load(f))
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

    args["device"] = device

    args["env"]["cfg"]["device"] = device
    args["env"]["cfg"]["seed"] = args["seed"]
    runner = Runner(device=device)
    runner.setup(args)
    if args.get("cdf_path") is not None:
        runner.load(cdf_path=args["cdf_path"])
    if args.get("policy_path") is not None:
        runner.load(policy_path=args["policy_path"])
    if args.get("value_path") is not None:
        runner.load(value_path=args["value_path"])
    if args.get("train", False): 
        if args.get("track", False):
            wandb.init(project=args.get("wandb_project", "hippo"),
                    sync_tensorboard=True,
                    monitor_gym=True,
                    save_code=True,
                    name=args.get("experiment_name", None),
                    config = {**args,"method":"hippo"})
        runner.train()
    if args.get("save", False):
        env_name = args["env"]["name"]
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        cdf_path = env_name + "_cdf" + now + ".pt"
        policy_path = env_name + "_policy" + now + ".pt"
        value_path = env_name + "_value" + now + ".pt"
        runner.save(cdf_path=cdf_path,
                    policy_path=policy_path,
                    value_path=value_path) 
    if args.get("eval", False):
        logs = runner.evaluate()
        print("Evaluation logs: ", logs)
    if args.get("visualize", False):
        runner.visualize_results()
    
    if wandb.run is not None:
        wandb.finish()