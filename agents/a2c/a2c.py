import torch
import logging
from torch import nn
from torch.distributions import Categorical
from agents.torch_utils import create_layered_network, ToMatrix
from agents.ql.ql import QFunction, QFunctionNetwork
from abc import ABC, abstractmethod

LOGGER = logging.getLogger("main.a2c")


class ValueNetwork(nn.Module):
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
        opponent_action_dim: int = None,
    ):
        super().__init__()
        consider_opponent_action = opponent_action_dim is not None
        
        self._consider_opponent_action = consider_opponent_action

        output_dim = action_dim * (opponent_action_dim if consider_opponent_action else 0)
        self.actor_layers = create_layered_network(obs_dim, output_dim, hidden_layer_sizes, hidden_layer_activation)
        if consider_opponent_action:
            self.actor_layers.append(ToMatrix(opponent_action_dim, action_dim))
        self.actor_layers.append(nn.Softmax(dim=-1))
        self.representation = nn.Identity() if representation is None else representation
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor):
        rep = self.representation(obs)

        return self.actor_layers(rep)
    
    def from_representation(self, rep: torch.Tensor) -> torch.Tensor:
        return self.actor_layers(rep)
    
    @property
    def consider_opponent_action(self) -> bool:
        return self._consider_opponent_action


class A2CLearnerBase(ABC):
    @abstractmethod
    def sample_action(self, obs: torch.Tensor, **kwargs) -> int:
        pass

    @abstractmethod
    def learn(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, **kwargs):
        pass

    @property
    @abstractmethod
    def actor(self) -> nn.Module | None:
        """The actor module, if it exists."""

    @property
    @abstractmethod
    def critic(self) -> nn.Module | None:
        """The critic module, if it exists."""


class A2CLambdaLearner(A2CLearnerBase):
    def __init__(
        self,
        actor: ActorNetwork,
        critic: ValueNetwork,
        discount: float = 1.0,
        actor_lambda: float = 0.0,
        critic_lambda: float = 0.0,
        actor_entropy_loss_coef: float = 0.0,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 1e-4,
        policy_improvement_steps: int = 1,
        policy_evaluation_steps: int = 1,
        policy_cumulative_discount: bool = True,
    ):
        """Implementation of the advantage actor-critic algorithm with eligibility traces, from the Sutton & Barto book"""
        self._actor = actor
        self._critic = critic
        self.discount = discount
        self.actor_lambda = actor_lambda
        self.critic_lambda = critic_lambda
        self.actor_entropy_loss_coef = actor_entropy_loss_coef
        self.policy_improvement_steps = policy_improvement_steps
        self.policy_evaluation_steps = policy_evaluation_steps
        self.policy_cumulative_discount = policy_cumulative_discount

        self.actor_traces = [
            torch.zeros_like(parameter)
            for parameter in actor.parameters()
        ]
        self.critic_traces = [
            torch.zeros_like(parameter)
            for parameter in critic.parameters()
        ]

        # Due to the way the gradients are set up, we want the optimizer to maximize (i.e., leave the gradients' sign unchanged)
        self.actor_optimizer = torch.optim.SGD(self._actor.parameters(), maximize=True, lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.SGD(self._critic.parameters(), maximize=True, lr=critic_learning_rate)

        self.action_distribution = None
        self.action = None

        # Variables for policy improvement
        self.at_policy_improvement = True
        # Track at which environment step we are since we last performed policy improvement/evaluation
        self.policy_iteration_step = 0

        # Discount throughout a single episode
        self.cumulative_discount = 1.0
        
        # Track values
        self.delta = 0.0

    def sample_action(self, obs: torch.Tensor, **kwargs) -> int:
        """Sample an action from the actor. A training step starts with `sample_action()`, followed immediately by an environment step and `learn()`."""    
        action_probabilities = self._actor(obs)
        self.action_distribution = Categorical(probs=action_probabilities)
        self.action = self.action_distribution.sample()
        
        return self.action.item()

    def _step_policy_iteration(self):
        step_threshold = self.policy_improvement_steps if self.at_policy_improvement else self.policy_evaluation_steps
        
        self.policy_iteration_step += 1
        if self.policy_iteration_step >= step_threshold:
            self.at_policy_improvement = not self.at_policy_improvement
            self.policy_iteration_step = 0

    def _update_critic(self, obs: torch.Tensor):
        self.critic_optimizer.zero_grad()

        critic_score = self._critic(obs)
        critic_score.backward()
        with torch.no_grad():
            for critic_trace, parameter in zip(self.critic_traces, self._critic.parameters()):
                critic_trace.copy_(self.discount * self.critic_lambda * critic_trace + parameter.grad)
                parameter.grad.copy_(self.delta * critic_trace)

        self.critic_optimizer.step()

    def _update_actor(self):
        self.actor_optimizer.zero_grad()

        actor_score = self.cumulative_discount * self.action_distribution.log_prob(self.action)
        actor_score.backward(retain_graph=True)
        with torch.no_grad():
            for actor_trace, parameter in zip(self.actor_traces, self._actor.parameters()):
                actor_trace.copy_(self.discount * self.actor_lambda * actor_trace + parameter.grad)
                parameter.grad.copy_(self.delta * actor_trace * (1 - self.actor_entropy_loss_coef))
        # Add the entropy score gradient
        entropy_score = self.actor_entropy_loss_coef * self.action_distribution.entropy()
        entropy_score.backward()

        self.actor_optimizer.step()

    def compute_advantage(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool) -> float:
        """Compute the TD delta (a.k.a. advantage)."""
        with torch.no_grad():
            if terminated:
                target = reward
            else:
                target = reward + self.discount * self._critic(next_obs)
            
            delta = (target - self._critic(obs)).item()
    
        return delta

    # TODO: if we use linear function approximation, could we do True Online TD(lambda)?
    def learn(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, **kwargs):
        """Update the actor and critic networks in this environment step. Should be preceded by an environment interaction with `sample_action()`."""
        # Compute the TD delta
        self.delta = self.compute_advantage(obs, next_obs, reward, terminated)

        # Update the networks.
        # We perform policy improvement first and then policy evaluation (at_policy_improvement begins True).
        if self.at_policy_improvement:
            self._update_actor()
            self._step_policy_iteration()
        # We don't perform elif since we may want to perform both policy improvement and evaluation in the same step (e.g. if policy_improvement_steps == 1)
        if not self.at_policy_improvement:
            self._update_critic(obs)
            self._step_policy_iteration()

        if self.policy_cumulative_discount:
            self.cumulative_discount *= self.discount

        if terminated or truncated:
            self.cumulative_discount = 1.0
            for actor_trace in self.actor_traces:
                actor_trace.zero_()
            for critic_trace in self.critic_traces:
                critic_trace.zero_()

    @property
    def actor(self):
        return self._actor

    @property
    def critic(self):
        return self._critic


# NOTE: we can use both players' perspectives when updating the Q-value function. This is left to the training loop to manage
class A2CQLearner(A2CLearnerBase):
    def __init__(
        self,
        actor: ActorNetwork,
        critic: QFunction,
        actor_entropy_loss_coef: float = 0.0,
        actor_learning_rate: float = 1e-4,
        policy_cumulative_discount: bool = True,
        update_style: str = "expected-sarsa",
    ):
        """Implementation of a custom actor-critic algorithm with a Q-value table for the critic"""
        if update_style not in ("sarsa", "expected-sarsa", "q-learning"):
            raise ValueError("'update_style' must be one of 'sarsa', 'expected-sarsa', or 'q-learning'")
        
        self._actor = actor
        self._critic = critic
        self.discount = critic.discount
        self.actor_entropy_loss_coef = actor_entropy_loss_coef
        self.policy_cumulative_discount = policy_cumulative_discount
        self.consider_opponent_action = actor.consider_opponent_action
        # TODO: use
        self.update_style = update_style

        self.action_dim = self._critic.action_dim
        self.opponent_action_dim = self._critic.opponent_action_dim

        # Due to the way the gradients are set up, we want the optimizer to maximize (i.e., leave the gradients' sign unchanged)
        self.actor_optimizer = torch.optim.SGD(self._actor.parameters(), lr=actor_learning_rate, maximize=True)

        self.action_probabilities = None
        self.action_distribution = None
        self.action = None
        self.postponed_learn: dict = None
        self.frameskipped_critic_updates = []
        self.frameskipped_critic_updates_cumulative_reward = 0.0

        # Discount throughout a single episode
        self.cumulative_discount = 1.0
        
        # Track values
        self.delta = 0.0
        self.td_error = 0.0

    def compute_action_probabilities(self, obs: torch.Tensor, next_opponent_action: int) -> torch.Tensor:
        """Get the action probability distribution for the given observation and predicted opponent action."""
        if next_opponent_action is None:
            next_opponent_action = slice(None)

        action_probabilities = self._actor(obs)[:, next_opponent_action, :]
        return action_probabilities

    def sample_action(self, obs: torch.Tensor, *, next_opponent_action: int, **kwargs) -> int:
        """Sample an action from the actor. A training step starts with `sample_action()`, followed immediately by an environment step and `learn()`."""    
        self.action_probabilities = self.compute_action_probabilities(obs, next_opponent_action)
        self.action_distribution = Categorical(probs=self.action_probabilities)
        self.action = self.action_distribution.sample()
        return self.action.item()

    def _update_actor(self, obs: torch.Tensor, agent_action: int, opponent_action: int):
        # Compute the TD delta
        delta = self.compute_advantage(obs, agent_action, opponent_action)
        self.delta = delta
        
        # Calculate the probability distribution at obs considering opponent action, and consider we did the given action
        action_probabilities = self.compute_action_probabilities(obs, opponent_action)
        action_distribution = Categorical(probs=action_probabilities)
        action_tensor = torch.tensor(agent_action, dtype=torch.int64)
        
        # Update the actor network
        self.actor_optimizer.zero_grad()

        actor_score = (1 - self.actor_entropy_loss_coef) * self.cumulative_discount * delta * action_distribution.log_prob(action_tensor) + self.actor_entropy_loss_coef * action_distribution.entropy()
        actor_score.backward()

        self.actor_optimizer.step()

        if LOGGER.isEnabledFor(logging.DEBUG):
            new_action_probabilities = self.compute_action_probabilities(obs, opponent_action)
            new_action_probabilities = new_action_probabilities.detach()
            new_action_distribution = Categorical(probs=new_action_probabilities)
            LOGGER.debug("Actor was updated using delta %s for action %s, going from a distribution of %s with entropy %s to a distribution of %s with entropy %s", delta, agent_action, action_probabilities, action_distribution.entropy().item(), new_action_probabilities, new_action_distribution.entropy().item())

    def compute_advantage(self, obs: torch.Tensor, agent_action: int, opponent_action: int) -> float:
        """Compute the TD delta (a.k.a. advantage)."""
        # If the opponent's action doesn't matter, since it's ineffectual, then compute the advantage considering all possible opponent actions.
        # NOTE: another possibility would be to perform random sampling? But then it would probably take a long time before convergence...
        if opponent_action is None:
            opponent_action = slice(None)

        # A(s, o, a) = Q(s, o, a) - V(s, o) = Q(s, o, a) - pi.T Q(s, o, .)
        q_soa = self._critic.q(obs, agent_action, opponent_action).detach()
        pi = self._actor(obs)[:, opponent_action, :].detach()
        q_so = self._critic.q(obs, opponent_action=opponent_action).detach()
        return q_soa - (pi @ q_so.T).item()
    
    def _learn_complete(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, obs_agent_action: int, obs_opponent_action: int, next_obs_agent_action: int, next_obs_opponent_action: int):
        """Perform a complete update. This method is not called by the training loop since in practice it needs to be performed one-step later as we need to know the agent's and opponent's actual action on the next observation `next_obs`."""
        # Update the Q-table. Save the TD error in case the caller wants to check it. The TD error is None if no critic update was performed
        self.td_error = None
        if obs_agent_action is not None:
            next_obs_action_probabilities = self.compute_action_probabilities(next_obs, next_opponent_action=None).detach().squeeze().T
        else:
            next_obs_action_probabilities = torch.ones(self.action_dim, self.opponent_action_dim).float() / (self.action_dim)
        self.frameskipped_critic_updates.append((obs, reward, next_obs, terminated, obs_agent_action, obs_opponent_action, next_obs_action_probabilities))
        if next_obs_agent_action is not None:
            # Perform the frameskipped updates assuming no discounting (that's pretty much what's special about frameskipped updates)
            previous_discount = self._critic.discount
            self._critic.discount = 1.0
            total_td_error = torch.tensor(0.0)

            # We perform the updates in reverse order so that the future reward is propagated back to the oldest retained updates
            for frameskipped_update in reversed(self.frameskipped_critic_updates):
                obs_, reward_, next_obs_, terminated_, obs_agent_action_, obs_opponent_action_, next_obs_action_probabilities_ = frameskipped_update
                total_td_error = total_td_error + self._critic.update(
                    obs=obs_,
                    reward=reward_,
                    next_obs=next_obs_,
                    terminated=terminated_,
                    agent_action=obs_agent_action_,
                    opponent_action=obs_opponent_action_,
                    next_opponent_action=None, # We ignore what the opponent actually did, so that we would not need importance sampling. This will assume that the opponent behaves uniformly randomly
                    next_agent_policy=next_obs_action_probabilities_,
                )
            
            self.td_error = total_td_error.mean().item()
            self._critic.discount = previous_discount
            self.frameskipped_critic_updates.clear()

        # If the agent action is None then that means the agent couldn't act, so it doesn't make sense to update the actor
        if obs_agent_action is None:
            pass
        # If the agent did perform an action, then update the actor
        else:
            # If the opponent is being frameskipped, then we consider them as doing any of their actions, and update accordingly
            if obs_opponent_action is None:
                # TODO: how about considering it to be 0? (i.e. STAND, the action of doing nothing) The problem is that it would be dependent on the opponent model correctly predicting STAND, which is probably not likely ;_;
                for opponent_action in range(self.opponent_action_dim):
                    self._update_actor(obs, obs_agent_action, opponent_action)
            # We have to recalculate the policy's action distribution since the opponent action is likely different
            else:
                self._update_actor(obs, obs_agent_action, obs_opponent_action)

        if self.policy_cumulative_discount:
            self.cumulative_discount *= self.discount

        if terminated or truncated:
            self.cumulative_discount = 1.0

    def learn(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, *, obs_agent_action: int, obs_opponent_action: int, **kwargs):
        """
        Update the actor and critic in this environment step. Should be preceded by an environment interaction with `sample_action()`.
        
        NOTE: actual learning occurs one-step later, since we need to know the opponent's actual action on the next observation

        Extra parameters
        ----------------
        - `obs_agent_action`: which action the agent actually performed. Should match what was sampled, but can be `None` if the critic update should be done for all agent actions
        - `obs_opponent_action`: which action the opponent actually performed. May not match what was used at the time of sampling
        """
        if self.postponed_learn is not None:
            self._learn_complete(**self.postponed_learn, next_obs_agent_action=obs_agent_action, next_obs_opponent_action=obs_opponent_action)
            
            if terminated or truncated:
                self.postponed_learn = None
            
        self.postponed_learn = {
            "obs": obs,
            "next_obs": next_obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "obs_agent_action": obs_agent_action,
            "obs_opponent_action": obs_opponent_action,
        }

    @property
    def actor(self):
        return self.actor
    
    @property
    def critic(self):
        if isinstance(self._critic, QFunctionNetwork):
            return self._critic.q_network
        return None
