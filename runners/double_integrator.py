from .base import BaseRunner
from envs.integrator import( 
    MultiObjectiveDoubleIntegratorEnv,
    plot_integrator_trajectories,
    plot_value_function_integrator
)
from models.factory import PolicyFactory, ValueFactory

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
import torch
from torch import nn
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    CatTensors,
    UnsqueezeTransform,
)
from results.evaluate import PolicyEvaluator, calculate_bellman_violation
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import wandb
from typing import Dict, Any


class DoubleIntegratorRunner(BaseRunner):
    def __init__(self, device:torch.device):
        super(DoubleIntegratorRunner, self).__init__(device)
        self.device = device
        self.env = None
        self.policy_module = None
        self.cdf_module = None
        self.value_module = None
        self.parameters = None
        self.evaluator = None
    def setup(self,args):
        self.args = args
        parameters = TensorDict(
            args["env_params"],
            batch_size=[],device=self.device
        )
        self.parameters = parameters
        base_env = MultiObjectiveDoubleIntegratorEnv(batch_size=args.get("num_parallel_env"),
                                                    device=self.device,
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
        self.env = TransformedEnv(
            base_env,
            Compose(
                *transforms
            )
        ).to(self.device)
        self.env.transform[3].init_stats(num_iter=1000,reduce_dim=(0,1),cat_dim=1)
        self.env.transform[4].init_stats(num_iter=1000,reduce_dim=(0,1),cat_dim=1)
        
        #######################
        # Models:
        #######################


        cdf_net_config = {
            "name": "feedforward",
            "eps": 1e-2,
            "layers": [64, 64],
            "activation": nn.ReLU(),
            "device": self.device,
            "input_size": len(obs_signals),
            "bounded": True,
        }
        value_net_config = {
            "name": "feedforward",
            "eps": 1e-2,
            "layers": [64, 64],
            "activation": nn.ReLU(),
            "device": self.device,
            "input_size": len(obs_signals+ref_signals),
            "bounded": True,
        }
        policy_net_config = {
            "name": "feedforward",
            "layers": [64, 64, 2*self.env.action_spec.shape[-1]],
            "activation": nn.ReLU(),
            "device": self.device,
            "input_size": len(ref_signals+obs_signals),
        }
        actor_net = PolicyFactory.create(**policy_net_config)
        self.policy_module = ProbabilisticActor(
            module=TensorDictModule(actor_net,
                                    in_keys=["obs_extended"],
                                    out_keys=["loc", "scale"]),
            in_keys=["loc", "scale"],
            spec=self.env.action_spec,
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": -parameters["max_input"],
                "high": parameters["max_input"],
            },
            return_log_prob=True,
        )

        value_net = ValueFactory.create(**value_net_config)
        self.value_module = ValueOperator(
            module=value_net,
            in_keys=["obs_extended"],
            out_keys=["V2"]
        )

        cdf_net = ValueFactory.create(**cdf_net_config)
        self.cdf_module = ValueOperator(
            module=cdf_net,
            in_keys=["obs"],
            out_keys=["V1"],
        )

        self.evaluator = PolicyEvaluator(env=self.env,
                                    policy_module=self.policy_module,
                                    rollout_len=self.args["max_rollout_len"],
                                    keys_to_log=[self.args.get("primary_reward_key"),
                                                self.args.get("secondary_reward_key"),
                                                "step_count"])
        
    def evaluate(self):
        if self.args == None:
            raise ValueError("Setup the runner before evaluating")
        logs = defaultdict(list)
        eval_logs = self.evaluator.evaluate_policy()
        logs.update(eval_logs)
        if self.args.get("track_bellman_violation",False):
            bm_viol, *_ = calculate_bellman_violation(
                self.args.get("bellman_eval_res",10),
                self.cdf_module,
                self.args["state_space"],
                self.policy_module,
                MultiObjectiveDoubleIntegratorEnv, 
                self.args.get("gamma"),
                transforms=self.env.transform[:-1]
            )
            logs["bellman_violation_mean"] = bm_viol.flatten().mean().item()
            logs["bellman_violation_max"] = bm_viol.flatten().max().item()
            logs["bellman_violation_std"] = bm_viol.flatten().std().item()
        return logs

    def plot_results(self):
        if self.args == None:
            raise ValueError("Setup the runner before plotting")
        if self.args.get("plot_CBF") and self.args.get("plot_traj") > 0:
            print("Plotting CBF")
            resolution = 10
            plot_value_function_integrator(self.parameters["max_x1"], 
                                        self.parameters["max_x2"],
                                        resolution,
                                        self.cdf_module,
                                        transforms=self.env.transform[:-1])
        if self.args.get("plot_traj") > 0:
            plot_integrator_trajectories(self.env, 
                                        self.policy_module,
                                        self.args["max_rollout_len"],
                                        self.args.get("plot_traj"),
                                        self.cdf_module)
            print("Plotted trajectories")
        if self.args.get("plot_bellman_violation"):
            print("Calculating and plotting Bellman violation")
            bm_viol,mesh = calculate_bellman_violation(10, 
                                                self.cdf_module,
                                                self.args["state_space"], 
                                                self.policy_module,
                                                MultiObjectiveDoubleIntegratorEnv,
                                                self.args.get("gamma"),
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