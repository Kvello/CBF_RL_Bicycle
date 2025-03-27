from torchrl.objectives.common import LossModule
from torchrl.objectives.ppo import ClipPPOLoss
from tensordict.nn import ProbabilisticTensorDictSequential, TensorDictModule
from tensordict import TensorDict, TensorDictBase
import torch

#TODO: Allow for modifiable tensordict keys in loss functions
class HiPPOLoss(LossModule):
    def __init__(
        self,
        actor: ProbabilisticTensorDictSequential,
        primary_critic: TensorDictModule,
        secondary_critic: TensorDictModule,
        primary_reward_key: str = "r1",
        secondary_reward_key: str = "r2",
        *,
        clip_epsilon: float = 0.2,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        CBF_critic_coef: float = 1.0,
        secondary_critic_coef: float = 1.0,
        safety_objective_coef: float = 1.0,
        secondary_objective_coef: float = 1.0,
        normalize_advantage: bool = False,
        gamma: float = None,
        separate_losses: bool = False,
        reduction: str = None,
        **kwargs,
    ):
        super().__init__()
        self.actor_network = actor
        self.primary_critic = primary_critic
        self.secondary_critic = secondary_critic
        self.primary_value_key = primary_critic.out_keys[0]
        self.secondary_value_key = secondary_critic.out_keys[0]
        self.primary_reward_key = primary_reward_key
        self.secondary_reward_key = secondary_reward_key

        self.CBF_critic_coef = CBF_critic_coef
        self.secondary_critic_coef = secondary_critic_coef
        self.safety_objective_coef = safety_objective_coef
        self.secondary_objective_coef = secondary_objective_coef

        # We only use these to calculate the objective losses
        self.primary_loss = ClipPPOLoss(
            actor_network=actor,
            critic_network=primary_critic,
            clip_epsilon=clip_epsilon,
            entropy_bonus=False,
            entropy_coef=0.0,
            critic_coef=self.CBF_critic_coef,
            loss_critic_type="smooth_l1",
        )
        self.primary_loss.set_keys(
            advantage="A1",
            value=primary_critic.out_keys[0],
            value_target="V1_target",
            reward=self.primary_reward_key,
        )
        self.secondary_loss = ClipPPOLoss(
            actor_network=actor,
            critic_network=secondary_critic,
            clip_epsilon=clip_epsilon,
            entropy_bonus=False,
            entropy_coef=0.0,
            critic_coef=self.secondary_critic_coef,
            loss_critic_type="smooth_l1",
        )
        self.secondary_loss.set_keys(
            advantage="A2",
            value=secondary_critic.out_keys[0],
            value_target="V2_target",
            reward=secondary_reward_key,
        )
    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss_safety_objective",
                    "loss_secondary_objective",
                    "loss_CBF",
                    "loss_secondary_critic"]
            keys.append("ESS")
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values
    
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        primary_value = self.primary_critic(tensordict)[self.primary_value_key]
        secondary_value = self.secondary_critic(tensordict)[self.secondary_value_key]

        with torch.no_grad():
            tensordict["td_error"] = torch.abs(tensordict["V1_target"] - primary_value)
        primary_critic_loss = torch.nn.HuberLoss(reduction='none')(primary_value, tensordict["V1_target"])
        secondary_critic_loss = torch.nn.HuberLoss(reduction='none')(secondary_value, tensordict["V2_target"])
        # Allow for importance sampling of the samples weigthed on the primary critic td error
        if "_weight" in tensordict:
            primary_critic_loss = (tensordict["_weight"] * primary_critic_loss).mean() 
            secondary_critic_loss = (tensordict["_weight"] * secondary_critic_loss).mean()
            # I think this is a valid way of weighting the actor loss
            tensordict["A1"] = tensordict["A1"]* tensordict["_weight"]
            tensordict["A2"] = tensordict["A2"]* tensordict["_weight"]
        else:
            primary_critic_loss = primary_critic_loss.mean()
            secondary_critic_loss = secondary_critic_loss.mean()
        secondary_loss_vals = self.secondary_loss(tensordict)
        primary_loss_vals = self.primary_loss(tensordict)
        primary_loss_vals["loss_critic"] = primary_critic_loss * self.CBF_critic_coef
        secondary_loss_vals["loss_critic"] = secondary_critic_loss * self.secondary_critic_coef
        primary_loss_vals["loss_objective"] = primary_loss_vals["loss_objective"] * self.safety_objective_coef
        secondary_loss_vals["loss_objective"] = secondary_loss_vals["loss_objective"] * self.secondary_objective_coef

        td_out = TensorDict(
            {
                "loss_safety_objective": primary_loss_vals["loss_objective"],
                "loss_secondary_objective": secondary_loss_vals["loss_objective"],
                "loss_CBF": primary_loss_vals["loss_critic"],
                "loss_secondary_critic": secondary_loss_vals["loss_critic"],
            }
        )
        return td_out