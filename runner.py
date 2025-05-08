from algorithms.hippo import HierarchicalPPO as HiPPO
from algorithms.ppo import PPO
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
from envs import make_env, SafetyGymEnv
from models.factory import PolicyFactory, ValueFactory
import safety_gym
from torchrl.envs.utils import step_mdp
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.utils import get_config_value
from warnings import warn
import imageio
import numpy as np

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
        if self.args["algorithm"]["name"] == "hippo":
            ppo_entity = HiPPO()
            algo_args = self.args["algorithm"]
            algo_args["device"] = self.device
            algo_args["gamma"] = self.args["env"]["gamma"]
            algo_args["safety_obs_key"] = self.args["env"]["cfg"]["safety_obs_key"]
            ppo_entity.setup(algo_args)
        elif self.args["algorithm"]["name"] == "ppo":
            ppo_entity = PPO()
            algo_args = self.args["algorithm"]
            algo_args["device"] = self.device
            algo_args["gamma"] = self.args["env"]["gamma"]
            algo_args["safety_obs_key"] = self.args["env"]["cfg"]["safety_obs_key"]
            ppo_entity.setup(algo_args)
        else:
            raise ValueError("Algorithm not supported")
        print("Training with config: ",self.args)
        if self.args.get("train",False):
            collector = SyncDataCollector(
                create_env_fn=self.env,
                policy=self.policy_module,
                frames_per_batch=algo_args["frames_per_batch"],
                total_frames=algo_args["total_frames"],
                split_trajs=False,
                device=self.device,
                exploration_type=ExplorationType.RANDOM,
                reset_at_each_iter=True)

            replay_buffer = TensorDictReplayBuffer(
                storage=LazyTensorStorage(max_size=algo_args["frames_per_batch"]),
                sampler=SamplerWithoutReplacement(),
            )
            optim = torch.optim.Adam
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optim, args["total_frames"] // args["frames_per_batch"], 1e-8
            # )
            if self.args["algorithm"]["name"] == "hippo":
                ppo_entity.train(
                    policy_module=self.policy_module,
                    V_primary=self.cdf_module,
                    V_secondary=self.value_module,
                    optim=optim,
                    collector=collector,
                    replay_buffer=replay_buffer,
                    eval_func=self.evaluate
                ) 
            elif self.args["algorithm"]["name"] == "ppo":
                ppo_entity.train(
                    policy_module=self.policy_module,
                    value_module=self.cdf_module,
                    optim=optim,
                    collector=collector,
                    replay_buffer=replay_buffer,
                    eval_func=self.evaluate
                )
            else:
                raise ValueError("Algorithm not supported")
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
                                    eval_steps=self.args["env"]["cfg"]["max_steps"],
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
                reward_key=self.args["algorithm"]["primary_reward_key"],
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
            state_dict = torch.load(value_path,map_location=self.device,weights_only=True)
            self.value_module.load_state_dict(state_dict)
            print("Value network loaded")       
        if policy_path is not None:
            state_dict = torch.load(policy_path,map_location=self.device,weights_only=True)
            self.policy_module.load_state_dict(state_dict)
            print("Policy loaded")
        if cdf_path is not None:
            state_dict = torch.load(cdf_path,map_location=self.device,weights_only=True)
            self.cdf_module.load_state_dict(state_dict)
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
        resolution = plotting_args.get("resolution", 100)
        if plotting_args.get("cdf",False):
            print("Plotting cdf")
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
            bm_viol,mesh = calculate_bellman_violation(resolution,
                                                self.cdf_module,
                                                plotting_args["state_space"], 
                                                self.policy_module,
                                                self.env_maker,
                                                self.args["env"]["gamma"],
                                                reward_key=self.args["algorithm"]["primary_reward_key"],
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
            cfg = self.args["env"]["cfg"]
            # rendering only works with unbatched GymEnvs
            cfg["num_parallel_env"] = 1
            env = make_env(self.args["env"]["name"],cfg)
            #resetting has to happen before camera id is obtained
            td = env.reset()

            warn_str = "Warning: rendering mode not set. Defaulting to record"
            render_mode = get_config_value(render_args, "mode", "record", warn_str)
            if render_mode not in ["human", "record"]:
                warn_str = f"Warning: Render mode {render_mode} not supported. Defaulting to 'record'"
                warn(warn_str)
                render_mode = "record"
            if render_mode == "record":
                frames = []
                warn_str = "Warning: camera not set. Defaulting to 'track'"
                camera = get_config_value(render_args, "camera", "track", warn_str)
                camera_id = env.camera_name2id(camera)
            warn_str = "Warning: fps not set. Defaulting to 60"
            fps = get_config_value(render_args, "fps", 60, warn_str)
            warn_str = "Warning: num_frames not set. Defaulting to 1000"
            num_frames = get_config_value(render_args, "num_frames", 1000, warn_str)
            
            mode = "rgb_array" if render_mode == "record" else render_mode
            dt = 1.0/fps
            for _ in range(num_frames):
                if render_mode == "record":
                    render_kwargs = render_args.get("render_kwargs", {})
                    frame = env.render(mode=mode, camera_id=camera_id,**render_kwargs)
                    frames.append(frame)
                else:
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
            now = datetime.now().strftime("%Y%m%d-%H%M%S")
            video_path = (
                self.args["env"]["name"] + "_" +
                self.args["algorithm"]["name"] + 
                "_video_" +
                now + ".mp4"
            )
            imageio.mimsave(video_path, frames, fps=fps)
            print("Video saved to: ", video_path)
            if render_mode == "record":
                if wandb.run is not None:
                    wandb.log({"video": wandb.Video(video_path, format="mp4")})
                

                

multiprocessing.set_start_method("spawn", force=True)
is_fork = multiprocessing.get_start_method(allow_none=True) == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)



@hydra.main(config_path="configs", config_name="default.yaml",version_base="1.3")
def main(cfg: DictConfig) -> None:    
    config_file_path = "./configs/" + cfg.config_file
    with open(config_file_path, "r") as f:
        args = yaml.safe_load(f)
    args.update(OmegaConf.to_container(cfg, resolve=True))
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
    if args.get("wandb", False):
        wandb.init(project=args.get("wandb_project", args["env"]["name"] + "-" + args["algorithm"]["name"]),
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
                name=args.get("experiment_name", None),
                config = {**args})
    weights_dir = args.get("weight_dir", "models/weights/")
    if args.get("cdf_path",None) is not None:
        runner.load(cdf_path=weights_dir + args["cdf_path"])
    if args.get("policy_path",None) is not None:
        runner.load(policy_path=weights_dir + args["policy_path"])
    if args.get("value_path",None) is not None:
        runner.load(value_path=weights_dir + args["value_path"])
    if args.get("train", False): 
        runner.train()
    if args.get("save", False):
        env_name = args["env"]["name"]
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        cdf_path ="models/weights/" + env_name + "_cdf_" + now + ".pt"
        policy_path ="models/weights/" + env_name + "_policy_" + now + ".pt"
        value_path ="models/weights/" + env_name + "_value_" + now + ".pt"
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
if __name__ == "__main__":
    main()