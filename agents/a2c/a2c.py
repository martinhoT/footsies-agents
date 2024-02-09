import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
from stable_baselines3 import PPO
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
        self.layers.append(nn.Softmax(dim=1))

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class A2CModule:
    """Training module implementing the one-step advantage actor-critic algorithm"""
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        discount: float = 1.0,
        actor_learning_rate: float = 1e-2,
        critic_learning_rate: float = 1e-2,
    ):
        self.discount = discount

        self.actor = actor
        self.critic = critic

        self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=actor_learning_rate, maximize=True)
        self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=critic_learning_rate, maximize=True)

        self.action_distribution = None
        self.action = None

        # Discount throughout a single episode
        self.cumulative_discount = 1.0

    def _obs_to_torch(self, obs: np.ndarray) -> torch.Tensor:
        return torch.tensor(obs, dtype=torch.float32).reshape((1, -1))

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
        next_obs = self._obs_to_torch(next_obs)

        with torch.no_grad():
            # TODO: if this is simplified (a single statement), then does it affect grad?
            if terminated:
                target = reward
            else:
                target = reward + self.discount * self.critic(next_obs)

            delta = target - self.critic(obs)
            
        # TODO: when training the critic we define the target as a function of the critic itself, but we don't take it into account when calculating the gradient.
        #       Maybe experiment in the future with allowing it to backpropagate? Relevant Sutton & Barto: page 202, second to last paragraph
        # TODO: super unstable, mean squared error works
        critic_score = delta * self.critic(obs)
        critic_score.backward()
        self.critic_optimizer.step()

        actor_score = self.cumulative_discount * delta * self.action_distribution.log_prob(self.action)
        actor_score.backward()
        self.actor_optimizer.step()

        self.cumulative_discount *= self.discount

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.critic(obs)

    def policy(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.actor(obs)
