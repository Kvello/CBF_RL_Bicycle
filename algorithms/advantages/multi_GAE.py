from torchrl.objectives.value import GAE
import torch
from tensordict.nn import TensorDictModuleBase, TensorDictModule
from tensordict import TensorDict

#TODO: Allow for modifiable tensordict keys in advantage functions
class MultiGAE(TensorDictModuleBase):
    """This is a simple way of combining two GAEs into one module, which creates
    an unified API for the step method of any training algorithm.
    Ideally however, this should be implemented as a complete TensorDictModule.
    """
    def __init__(self, 
                 gamma:float, 
                 lmbda1:float,
                 lmbda2:float,
                 V_primary:TensorDictModule,
                 V_secondary:TensorDictModule,
                 device: torch.device=torch.device("cpu"),
                 primary_reward_key="r1",
                 secondary_reward_key="r2"):
        """Initializes the MultiGAE module

        Args:
            gamma (float): Discount factor of the MDP
            lmdba (float): lambda in the GAE algorithm
            V_primary (TensorDictModule): Primary value network
            V_secondary (TensorDictModule): Secondary value network
            device (torch.device, optional): device the networks are on. Defaults to torch.device("cpu").
            primary_reward_key (str, optional): Reward key of primary objective. Defaults to "r1".
            secondary_reward_key (str, optional): Reward kwy of secondary objective. Defaults to "r2".
        """

        super().__init__() 
        self.gamma = gamma
        self.lmbda1 = lmbda1
        self.lmbda2 = lmbda2
        self.device = device
        self.primary_reward_key = primary_reward_key
        self.secondary_reward_key = secondary_reward_key
        self.primary_advantage_key = "A1"
        self.secondary_advantage_key = "A2"
        self.primary_value_target_key = "V1_target"
        self.secondary_value_target_key = "V2_target"
        self.V_primary = V_primary
        self.V_secondary = V_secondary

        self.A_primary = GAE(
            gamma=self.gamma,
            lmbda=self.lmbda1,
            average_gae=False,
            value_network=self.V_primary,
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
            lmbda=self.lmbda2,
            average_gae=False,
            value_network=self.V_secondary,
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