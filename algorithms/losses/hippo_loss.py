from torchrl.objectives.common import LossModule
from torchrl.objectives.ppo import ClipPPOLoss
from tensordict.nn import ProbabilisticTensorDictSequential, TensorDictModule
from tensordict import TensorDict, TensorDictBase
import torch
from torchrl.collectors.utils import split_trajectories

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
            entropy_coef=entropy_coef, # No entropy_coef for safety objective
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
        self.primary_objective_loss = torch.tensor(0.0, device=primary_critic.device)
    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss_safety_objective",
                    "loss_secondary_objective",
                    "loss_CDF",
                    "loss_CDF_supervised"
                    "loss_secondary_critic",
                    "loss_secondary_entropy",
                    "loss_safety_entropy"]
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values
    
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        secondary_loss_vals = self.secondary_loss(tensordict)
        primary_loss_vals = self.primary_loss(tensordict)
        primary_objective_loss = self._calculate_primary_objective_loss(tensordict)
        primary_loss_vals["loss_objective"] = primary_objective_loss
        if "collision_states" in tensordict:
            collision_states = tensordict["collision_states"]
            CDF_collision_pred = self.primary_critic.module(collision_states)
            supervised_CDF_loss = (
                torch.nn.MSELoss(reduction="mean")(
                    CDF_collision_pred,
                    tensordict["collision_value"],
                )
            )*self.supervision_coef
        else:
            supervised_CDF_loss = torch.tensor(0.0, device=tensordict.device)
        assert not torch.isnan(
            primary_loss_vals["loss_objective"]
        ).any(), "NaN in primary loss"
        assert not torch.isnan(
            secondary_loss_vals["loss_objective"]
        ).any(), "NaN in secondary loss"
        assert not torch.isnan(
            primary_loss_vals["loss_critic"]
        ).any(), "NaN in primary critic loss"
        assert not torch.isnan(
            secondary_loss_vals["loss_critic"]
        ).any(), "NaN in secondary critic loss"
        assert not torch.isnan(
            primary_loss_vals["loss_entropy"]
        ).any(), "NaN in primary entropy loss"
        assert not torch.isnan(
            secondary_loss_vals["loss_entropy"]
        ).any(), "NaN in secondary entropy loss"
        td_out = TensorDict(
            {
                "loss_safety_objective": primary_loss_vals["loss_objective"],
                "loss_secondary_objective": secondary_loss_vals["loss_objective"],
                "loss_CDF": primary_loss_vals["loss_critic"],
                "loss_CDF_supervised": supervised_CDF_loss,
                "loss_secondary_critic": secondary_loss_vals["loss_critic"],
                "loss_secondary_entropy": secondary_loss_vals["loss_entropy"],
                "loss_safety_entropy": primary_loss_vals["loss_entropy"],
            }
        )
        return td_out
    def _calculate_primary_objective_loss(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        """
        Calculate the primary objective loss for the HiPPO algorithm.

        Args:
            tensordict (TensorDictBase): The input tensor dictionary containing all
                rollout data(not just the current batch).

        Returns:
            TensorDictBase: The tensor dictionary containing the primary objective loss.
        """
        # Extract transitions where the agent vioaltes the CDF constraint
        states = tensordict[self.primary_critic.in_keys[0]]
        next_states = tensordict["next"][self.primary_critic.in_keys[0]]
        state_CDF_values = self.primary_critic.module(states).squeeze(-1)
        next_state_CDF_values = self.primary_critic.module(next_states).squeeze(-1)
        # Rejection sampling:
        mask = (next_state_CDF_values < 0).bool()
        if mask.sum() == 0:
            # No transitions where the agent violates the CDF constraint
            # Return a loss of 0
            dummy = self.actor_network(tensordict)  # Just get something on the same graph
            return torch.zeros(1, device=tensordict.device) * dummy["action"].sum()
        data = tensordict[mask]
        loss_vals = self.primary_loss(data)
        return loss_vals["loss_objective"]