from abc import ABC, abstractmethod
from algorithms.hippo import HierarchicalPPO as HiPPO
from typing import Dict, Any
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.utils import ExplorationType
from typing import Optional
import torch
class BaseRunner():
    def __init__(self, device):
        self.device = device
        self.env = None
        self.policy_module = None
        self.cdf_module = None
        self.value_module = None
        self.args = None
    @abstractmethod
    def setup(self, args:Dict[str, Any]):
        pass
    def train(self):
        if self.args == None:
            raise ValueError("Setup the runner before training")
        ppo_entity = HiPPO()
        ppo_entity.setup(self.args)
        print("Training...")
        if self.args.get("train"):
            collector = SyncDataCollector(
                create_env_fn=self.env,
                policy=self.policy_module,
                frames_per_batch=self.args["frames_per_batch"],
                total_frames=self.args["total_frames"],
                split_trajs=False,
                device=self.device,
                exploration_type=ExplorationType.RANDOM)

            replay_buffer = TensorDictReplayBuffer(
                storage=LazyTensorStorage(max_size=self.args["frames_per_batch"]),
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
    def save(self,
             cdf_path:Optional[str]=None, 
             policy_path:Optional[str]=None, 
             value_path:Optional[str]=None):
        if self.args is None:
            raise ValueError("Setup the runner before saving")
        if cdf_path is None:
            torch.save(self.cdf_module.state_dict(), cdf_path)
        if policy_path is None:
            torch.save(self.policy_module.state_dict(),policy_path)
        if value_path is None:
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
    @abstractmethod
    def evaluate(self):
        pass
    @abstractmethod
    def plot_results(self):
        pass