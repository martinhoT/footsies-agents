import torch
from torch import nn


# What model to report with the `model` property of the agent, since it implements FootsiesAgentTorch
class FullModel(nn.Module):
    def __init__(
        self,
        game_model: nn.Module,
        opponent_model: nn.Module,
        actor_critic: nn.Module,
    ):
        super().__init__()

        self.game_model = game_model
        self.opponent_model = opponent_model
        self.actor_critic = actor_critic
    
    def forward(self, obs: torch.Tensor, agent_action_onehot: torch.Tensor, opponent_action_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        next_obs_representation = self.game_model(obs, agent_action_onehot, opponent_action_onehot) if self.game_model is not None else None
        opponent_action_probabilities = self.opponent_model(obs) if self.opponent_model is not None else None
        agent_action_probabilities, obs_value = self.actor_critic(obs) if self.actor_critic is not None else None

        return next_obs_representation, opponent_action_probabilities, agent_action_probabilities, obs_value
        