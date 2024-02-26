import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
from agents.torch_utils import create_layered_network
from agents.utils import extract_sub_kwargs


class CriticNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: nn.Module = nn.Identity,
        representation: nn.Module = None,
    ):
        super().__init__()

        self.critic_layers = create_layered_network(obs_dim, 1, hidden_layer_sizes, hidden_layer_activation)
        self.representation = nn.Identity() if representation is None else representation

    def forward(self, obs: torch.Tensor):
        rep = self.representation(obs)

        return self.critic_layers(rep)

    def from_representation(self, rep: torch.Tensor) -> torch.Tensor:
        return self.critic_layers(rep)


class ActorNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: nn.Module = nn.Identity,
        representation: nn.Module = None,
    ):
        super().__init__()

        self.actor_layers = create_layered_network(obs_dim, action_dim, hidden_layer_sizes, hidden_layer_activation)
        self.actor_layers.append(nn.Softmax(dim=1))
        self.representation = nn.Identity() if representation is None else representation
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor):
        rep = self.representation(obs)

        return self.actor_layers(rep)
    
    def from_representation(self, rep: torch.Tensor) -> torch.Tensor:
        return self.actor_layers(rep)


class A2CLambdaLearner:
    def __init__(
        self,
        actor: ActorNetwork,
        critic: CriticNetwork,
        discount: float = 1.0,
        actor_lambda: float = 0.0,
        critic_lambda: float = 0.0,
        actor_entropy_loss_coef: float = 0.0,
        actor_optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        critic_optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        **kwargs,
    ):
        """Implementation of the advantage actor-critic algorithm with eligibility traces, from the Sutton & Barto book"""
        self.actor = actor
        self.critic = critic
        self.discount = discount
        self.actor_lambda = actor_lambda
        self.critic_lambda = critic_lambda
        self.actor_entropy_loss_coef = actor_entropy_loss_coef

        self.actor_traces = [
            torch.zeros_like(parameter)
            for parameter in actor.parameters()
        ]
        self.critic_traces = [
            torch.zeros_like(parameter)
            for parameter in critic.parameters()
        ]

        actor_optimizer_kwargs, critic_optimizer_kwargs = extract_sub_kwargs(kwargs, ("actor_optimizer", "critic_optimizer"))

        actor_optimizer_kwargs = {
            "lr": 1e-4,
            **actor_optimizer_kwargs,
        }
        critic_optimizer_kwargs = {
            "lr": 1e-4,
            **critic_optimizer_kwargs,
        }

        # Due to the way the gradients are set up, we want the optimizer to maximize (i.e., leave the gradients unchanged)
        self.actor_optimizer = actor_optimizer(self.actor.parameters(), maximize=True, **actor_optimizer_kwargs)
        self.critic_optimizer = critic_optimizer(self.critic.parameters(), maximize=True, **critic_optimizer_kwargs)

        self.action_distribution = None
        self.action = None

        # Discount throughout a single episode
        self.cumulative_discount = 1.0
        
        # Track values
        self.delta = 0.0

    def _obs_to_torch(self, obs: np.ndarray) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            return torch.from_numpy(obs).float().reshape((1, -1))
        return obs

    def sample_action(self, obs: np.ndarray) -> int:
        """Sample an action from the actor. A training step starts with `act()`, followed immediately by an environment step and `update()`"""
        obs = self._obs_to_torch(obs)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        action_probabilities = self.actor(obs)
        self.action_distribution = Categorical(probs=action_probabilities)
        self.action = self.action_distribution.sample()
        
        return self.action.item()

    # TODO: if we use linear function approximation, could we do True Online TD(lambda)?
    def learn(self, obs: np.ndarray, next_obs: np.ndarray, reward: float, terminated: bool):
        """Update the actor and critic networks in this environment step. Should be preceded by an environment interaction with `act()`"""
        obs = self._obs_to_torch(obs)

        # Compute the TD delta
        with torch.no_grad():
            if terminated:
                target = reward
            else:
                next_obs = self._obs_to_torch(next_obs)
                target = reward + self.discount * self.critic(next_obs)
            
            delta = (target - self.critic(obs)).item()

        self.delta = delta

        # The actor should be updated first, since it already started training in act()
        actor_score = self.cumulative_discount * self.action_distribution.log_prob(self.action)
        actor_score.backward(retain_graph=True)
        with torch.no_grad():
            for actor_trace, parameter in zip(self.actor_traces, self.actor.parameters()):
                actor_trace.copy_(self.discount * self.actor_lambda * actor_trace + parameter.grad)
                parameter.grad.copy_(delta * actor_trace * (1 - self.actor_entropy_loss_coef))
        # Add the entropy score gradient
        entropy_score = self.actor_entropy_loss_coef * self.action_distribution.entropy()
        entropy_score.backward()

        critic_score = self.critic(obs)
        critic_score.backward()
        with torch.no_grad():
            for critic_trace, parameter in zip(self.critic_traces, self.critic.parameters()):
                critic_trace.copy_(self.discount * self.critic_lambda * critic_trace + parameter.grad)
                parameter.grad.copy_(delta * critic_trace)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.cumulative_discount *= self.discount

        if terminated:
            self.cumulative_discount = 1.0
            for actor_trace in self.actor_traces:
                actor_trace.zero_()
            for critic_trace in self.critic_traces:
                critic_trace.zero_()


class ImitationLearner:
    def __init__(
        self,
        policy: nn.Module,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        **kwargs,
    ):
        """Imitation learning applied to a policy"""
        optimizer_kwargs = extract_sub_kwargs(kwargs, ("optimizer",))
        
        self.policy = policy
        self.optimizer = optimizer(self.policy.parameters(), maximize=False, **optimizer_kwargs)
        self.loss_fn = nn.KLDivLoss(reduction="batchmean")

    def learn(self, obs: np.ndarray, action: int, frozen_representation: bool = False):
        """Update policy by imitating the action at the given observation"""
        obs = self._obs_to_torch(obs)
        action_onehot = torch.nn.functional.one_hot(torch.Tensor(action), num_classes=self.actor.action_dim)
        self.optimizer.zero_grad()

        if frozen_representation:
            with torch.no_grad():
                rep = self.actor.representation(obs)
            probs = self.actor.from_representation(rep)

        else:
            probs = self.actor(obs)

        loss = self.loss_fn(probs, action_onehot)
        loss.backward()

        self.optimizer.step()