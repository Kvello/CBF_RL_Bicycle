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
        entropy_coef: float = 0.0,
        samples_mc_entropy: int = 1,
        critic_coef: float = 1.0,
        supervision_coef: float = 1.0,
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
        self.primary_reward_key = primary_reward_key
        self.secondary_reward_key = secondary_reward_key
        self.critic_coef = critic_coef
        self.supervision_coef = supervision_coef

        # We only use these to calculate the objective losses
        self.primary_loss = ClipPPOLoss(
            actor_network=actor,
            critic_network=primary_critic,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_coef),
            entropy_coef=entropy_coef,
            critic_coef=self.critic_coef,
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
            entropy_bonus=bool(entropy_coef),
            entropy_coef=entropy_coef,
            critic_coef=self.critic_coef,
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
                    "loss_CBF_supervised"
                    "loss_secondary_critic",
                    "loss_safety_entropy",
                    "loss_secondary_entropy"]
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values
    
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        secondary_loss_vals = self.secondary_loss(tensordict)
        primary_loss_vals = self.primary_loss(tensordict)

        if "collision_states" in tensordict:
            collision_states = tensordict["collision_states"]
            CBF_collision_pred = self.primary_critic.module(collision_states)
            supervised_CBF_loss = (
                torch.nn.MSELoss(reduction="mean")(
                    CBF_collision_pred,
                    tensordict["collision_value"],
                )
            )*self.supervision_coef
        else:
            supervised_CBF_loss = torch.tensor(0.0, device=tensordict.device)
        td_out = TensorDict(
            {
                "loss_safety_objective": primary_loss_vals["loss_objective"],
                "loss_secondary_objective": secondary_loss_vals["loss_objective"],
                "loss_CBF": primary_loss_vals["loss_critic"],
                "loss_CBF_supervised": supervised_CBF_loss,
                "loss_secondary_critic": secondary_loss_vals["loss_critic"],
                "loss_safety_entropy": primary_loss_vals["loss_entropy"],
                "loss_secondary_entropy": secondary_loss_vals["loss_entropy"],
            }
        )
        return td_out