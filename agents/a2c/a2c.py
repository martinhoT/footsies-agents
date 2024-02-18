import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
from agents.torch_utils import create_layered_network


class CriticNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: nn.Module = nn.Identity,
    ):
        super().__init__()

        self.layers = create_layered_network(obs_dim, 1, hidden_layer_sizes, hidden_layer_activation)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class ActorNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: nn.Module = nn.Identity,
    ):
        super().__init__()

        self.layers = create_layered_network(obs_dim, action_dim, hidden_layer_sizes, hidden_layer_activation)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        y = self.layers(x)
        return self.softmax(y)


class A2CModule(nn.Module):
    """Training module implementing the advantage actor-critic algorithm with eligibility traces, from the Sutton & Barto book"""
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        discount: float = 1.0,
        actor_learning_rate: float = 1e-2,
        critic_learning_rate: float = 1e-2,
        actor_eligibility_traces_decay: float = 0.0,
        critic_eligibility_traces_decay: float = 0.0,
        actor_entropy_loss_coef: float = 0.0,
        optimizer: torch.optim.Optimizer = torch.optim.SGD,
    ):
        super().__init__()

        self.discount = discount
        self.actor_eligibility_traces_decay = actor_eligibility_traces_decay
        self.critic_eligibility_traces_decay = critic_eligibility_traces_decay
        self.actor_entropy_loss_coef = actor_entropy_loss_coef

        self.actor = actor
        self.critic = critic

        self.actor_traces = [
            torch.zeros(parameter.shape)
            for parameter in actor.parameters()
        ]
        self.critic_traces = [
            torch.zeros(parameter.shape)
            for parameter in critic.parameters()
        ]

        # Due to the way the gradients are set up, we want the optimizer to maximize (i.e., leave the gradients unchanged)
        self.actor_optimizer = optimizer(self.actor.parameters(), lr=actor_learning_rate, maximize=True)
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=critic_learning_rate, maximize=True)

        self.action_distribution = None
        self.action = None

        # Discount throughout a single episode
        self.cumulative_discount = 1.0
        
        # Track values
        self.delta = 0.0

    def _obs_to_torch(self, obs: np.ndarray) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            return torch.tensor(obs, dtype=torch.float32).reshape((1, -1))
        return obs

    def act(self, obs: np.ndarray) -> int:
        """Sample an action from the actor. A training step starts with `act()`, followed immediately by an environment step and `update()`"""
        obs = self._obs_to_torch(obs)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        action_probabilities = self.actor(obs)
        self.action_distribution = Categorical(action_probabilities)
        self.action = self.action_distribution.sample()
        
        return self.action.item()

    def update(self, obs: np.ndarray, next_obs: np.ndarray, reward: float, terminated: bool):
        """Update the actor and critic networks in this environment step. Should be preceded by an environment interaction with `act()`"""
        obs = self._obs_to_torch(obs)

        with torch.no_grad():
            # NOTE: watch out for when the episode ends, since bootstraping is not performed we don't have an equalizer in case the critic's values are exploding, which should translate in large loss/score and thus large gradients
            # TODO: if this is simplified (a single statement), then does it affect grad?
            if terminated:
                target = reward
            else:
                next_obs = self._obs_to_torch(next_obs)
                target = reward + self.discount * self.critic(next_obs)
            
            delta = (target - self.critic(obs)).item()

        self.delta = delta

        critic_score = self.critic(obs)
        critic_score.backward()
        with torch.no_grad():
            for critic_trace, parameter in zip(self.critic_traces, self.critic.parameters()):
                critic_trace.copy_(self.discount * self.critic_eligibility_traces_decay * critic_trace + parameter.grad)
                parameter.grad.copy_(delta * critic_trace)
        self.critic_optimizer.step()

        actor_score = (1 - self.actor_entropy_loss_coef) * self.cumulative_discount * self.action_distribution.log_prob(self.action) + self.actor_entropy_loss_coef * self.action_distribution.entropy()
        actor_score.backward()
        with torch.no_grad():
            for actor_trace, parameter in zip(self.actor_traces, self.actor.parameters()):
                actor_trace.copy_(self.discount * self.critic_eligibility_traces_decay * actor_trace + parameter.grad)
                parameter.grad.copy_(delta * actor_trace)
        self.actor_optimizer.step()

        self.cumulative_discount = 1.0 if terminated else (self.cumulative_discount * self.discount)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.critic(obs)

    def policy(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.actor(obs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value
