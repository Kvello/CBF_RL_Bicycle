from torchrl.objectives.value import GAE, ValueEstimatorBase
import torch
from tensordict.nn import TensorDictModule
from tensordict import TensorDict

class MultiGAE(ValueEstimatorBase):
    def __init__(self, 
                 gamma:float, 
                 lmdba:float,
                 V_primary:TensorDictModule,
                 V_secondary:TensorDictModule,
                 device: torch.device=torch.device("cpu"),
                 primary_reward_key="r1",
                 secondary_reward_key="r2",
                 primary_advantage_key="A1",
                 secondary_advantage_key="A2",
                 primary_value_target_key="V1_target",
                 secondary_value_target_key="V2_target"):
        self.gamma = gamma
        self.lmbda = lmdba
        self.device = device
        self.primary_reward_key = primary_reward_key
        self.secondary_reward_key = secondary_reward_key
        self.primary_advantage_key = primary_advantage_key
        self.secondary_advantage_key = secondary_advantage_key
        self.primary_value_target_key = primary_value_target_key
        self.secondary_value_target_key = secondary_value_target_key

        self.A_primary = GAE(
            gamma=self.gamma,
            lmbda=self.lmbda,
            value_network=self.V_primary,
            average_gae=True,
            device=self.device,
        )
        self.A_primary.set_keys(
            reward=self.primary_reward_key,
            advantage=self.primary_advantage_key,
            value_target=self.primary_value_target_key,
            value=V_primary.out_keys[0]
        )
        self.A_secondary = GAE(
            gamma=self.gamma,
            lmbda=self.lmbda,
            value_network=self.V_secondary,
            average_gae=True,
            device=self.device,
        )
        self.A_secondary.set_keys(
            reward=self.secondary_reward_key,
            advantage=self.secondary_advantage_key,
            value_target=self.secondary_value_target_key,
            value=V_secondary.out_keys[0]
        )

    def forward(self, tensordict_data:TensorDict):
        self.A_primary(tensordict_data)
        self.A_secondary(tensordict_data)
        return tensordict_data