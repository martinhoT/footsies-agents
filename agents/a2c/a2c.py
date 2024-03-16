from typing import Callable
import torch
import logging
from torch import nn
from torch.distributions import Categorical
from agents.torch_utils import create_layered_network, ToMatrix
from agents.ql.ql import QFunction
from abc import ABC, abstractmethod
from enum import Enum

LOGGER = logging.getLogger("main.a2c")


class ValueNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: type[nn.Module] = nn.Identity,
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
        hidden_layer_activation: type[nn.Module] = nn.Identity,
        representation: nn.Module = None,
        opponent_action_dim: int = None,
    ):
        super().__init__()
        if opponent_action_dim is None:
            raise NotImplementedError("not considering opponent actions is not supported yet")
    
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._hidden_layer_sizes = hidden_layer_sizes
        self._hidden_layer_activation = hidden_layer_activation
        self._representation = nn.Identity() if representation is None else representation
        self._opponent_action_dim = opponent_action_dim

        consider_opponent_action = opponent_action_dim is not None
        
        self._consider_opponent_action = consider_opponent_action

        output_dim = action_dim * (opponent_action_dim if consider_opponent_action else 0)
        self.actor_layers = create_layered_network(obs_dim, output_dim, hidden_layer_sizes, hidden_layer_activation)
        if consider_opponent_action:
            self.actor_layers.append(ToMatrix(opponent_action_dim, action_dim))
        self.actor_layers.append(nn.Softmax(dim=-1))

    def forward(self, obs: torch.Tensor):
        rep = self._representation(obs)

        return self.actor_layers(rep)
    
    def from_representation(self, rep: torch.Tensor) -> torch.Tensor:
        return self.actor_layers(rep)
    
    @property
    def consider_opponent_action(self) -> bool:
        return self._consider_opponent_action
    
    def probabilities(self, obs: torch.Tensor, next_opponent_action: int | None) -> torch.Tensor:
        """Get the action probability distribution for the given observation and predicted opponent action."""
        if next_opponent_action is None:
            next_opponent_action = slice(None)

        action_probabilities = self(obs)[:, next_opponent_action, :]
        return action_probabilities

    def sample_action(self, obs: torch.Tensor, next_opponent_action: int) -> torch.Tensor:
        """Randomly sample an action."""
        action_probabilities = self.probabilities(obs, next_opponent_action)
        action_distribution = Categorical(probs=action_probabilities)
        self.action = action_distribution.sample()
        return self.action

    def clone(self) -> "ActorNetwork":
        """Create a clone of this actor network, useful for extracting a policy."""
        if not isinstance(self._representation, nn.Identity):
            raise NotImplementedError("cloning actor networks with a non-identity representation is not supported yet")

        cloned = ActorNetwork(
            self._obs_dim,
            self._action_dim,
            hidden_layer_sizes=self._hidden_layer_sizes,
            hidden_layer_activation=self._hidden_layer_activation,
            representation=self._representation,
            opponent_action_dim=self._opponent_action_dim,
        )

        cloned.load_state_dict(self.state_dict())

        return cloned


class A2CLearnerBase(ABC):
    @abstractmethod
    def sample_action(self, obs: torch.Tensor, **kwargs) -> int:
        pass

    @abstractmethod
    def learn(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, **kwargs):
        pass

    @property
    @abstractmethod
    def actor(self) -> ActorNetwork:
        """The actor object."""

    @property
    @abstractmethod
    def critic(self) -> ValueNetwork | QFunction:
        """The critic object."""


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
    class UpdateStyle(Enum):
        SARSA = 0
        EXPECTED_SARSA = 1
        Q_LEARNING = 2
    
    def __init__(
        self,
        actor: ActorNetwork,
        critic: QFunction,
        actor_entropy_loss_coef: float = 0.0,
        actor_learning_rate: float = 1e-4,
        policy_cumulative_discount: bool = False,
        update_style: UpdateStyle = UpdateStyle.EXPECTED_SARSA,
        intrinsic_critic: QFunction = None,
    ):
        """
        Implementation of a custom actor-critic algorithm with a Q-value table/network for the critic
        
        Parameters
        ----------
        - `actor`: the actor (policy network)
        - `critic`: the critic (Q-value table/network)
        - `actor_entropy_loss_coef`: the coefficient for the entropy loss in the actor's loss function, i.e. how much to prioritize policy entropy over reward
        - `actor_learning_rate`: the learning rate for the actor
        - `policy_cumulative_discount`: whether to discount the policy more the more steps are taken in the environment
        - `update_style`: the style of update to use for the critic
        - `intrinsic_critic`: the critic for the intrinsic reward. If `None`, then no intrinsic reward is considered
        """
        if not isinstance(update_style, self.UpdateStyle):
            raise ValueError(f"update_style must be an instance of {self.UpdateStyle}, not {type(update_style)}")

        self._actor = actor
        self._critic = critic
        self._intrinsic_critic = intrinsic_critic
        self.actor_entropy_loss_coef = actor_entropy_loss_coef
        self.policy_cumulative_discount = policy_cumulative_discount
        self.consider_opponent_action = actor.consider_opponent_action
        self._update_style = update_style

        self.action_dim = self._critic.action_dim
        self.opponent_action_dim = self._critic.opponent_action_dim

        # Due to the way the gradients are set up, we want the optimizer to maximize (i.e., leave the gradients' sign unchanged)
        self.actor_optimizer = torch.optim.SGD(self._actor.parameters(), lr=actor_learning_rate, maximize=True)

        self.action = None
        self.postponed_learn: dict = None
        self.frameskipped_critic_updates = []
        self.frameskipped_critic_updates_cumulative_reward = 0.0
        
        # Consider a custom opponent policy when updating the Q-values.
        # Since the opponent is part of the environment, this pretty much defines the transition dynamics
        # and as such what kind of values we will obtain.
        self._custom_opponent_policy = None

        # Discount throughout a single episode
        self.cumulative_discount = 1.0
        
        # Track values
        self.delta = 0.0
        self.td_error = 0.0

    def sample_action(self, obs: torch.Tensor, *, next_opponent_action: int, **kwargs) -> int:
        """Sample an action from the actor. A training step starts with `sample_action()`, followed immediately by an environment step and `learn()`."""    
        return self.actor.sample_action(obs, next_opponent_action=next_opponent_action).item()

    def _update_actor(self, obs: torch.Tensor, agent_action: int, opponent_action: int | None):
        # Compute the TD delta
        delta = self.compute_advantage(obs, agent_action, opponent_action)
        self.delta = delta.mean().item() # We might get a delta vector
        
        # Calculate the probability distribution at obs considering opponent action, and consider we did the given action
        action_probabilities = self.actor.probabilities(obs, opponent_action)
        action_log_probabilities = torch.log(action_probabilities + 1e-8)
        
        # Update the actor network
        self.actor_optimizer.zero_grad()

        actor_entropy = -torch.sum(action_log_probabilities * action_probabilities, dim=-1)
        actor_delta = self.cumulative_discount * delta * action_log_probabilities[..., agent_action]
        actor_score = torch.mean((1 - self.actor_entropy_loss_coef) * actor_delta + self.actor_entropy_loss_coef * actor_entropy)
        actor_score.backward()

        self.actor_optimizer.step()

        if LOGGER.isEnabledFor(logging.DEBUG):
            new_action_probabilities = self.actor.probabilities(obs, opponent_action).detach()
            new_entropy = -torch.sum(torch.log(new_action_probabilities + 1e-8) * new_action_probabilities, dim=-1)
            LOGGER.debug("Actor was updated using delta %s for action %s, going from a distribution of %s with entropy %s to a distribution of %s with entropy %s", delta, agent_action, action_probabilities, actor_entropy, new_action_probabilities, new_entropy)

    def _compute_advantage_with_critic(self, obs: torch.Tensor, agent_action: int, opponent_action: int | None, critic: QFunction) -> torch.Tensor:
        """Compute the advantage with the supplied `critic`."""
        # If the opponent's action doesn't matter, since it's ineffectual, then compute the advantage considering all possible opponent actions.
        # NOTE: another possibility would be to perform random sampling? But then it would probably take a long time before convergence...
        if opponent_action is None:
            opponent_action = slice(None)

        # A(s, o, a) = Q(s, o, a) - V(s, o) = Q(s, o, a) - pi.T Q(s, o, .)
        q_soa = self._critic.q(obs, agent_action, opponent_action).detach()
        pi = self._actor(obs)[:, opponent_action, :].detach()
        q_so = self._critic.q(obs, opponent_action=opponent_action).detach()

        # This Einstein summation equation computes the diagonal of a >2D tensor by considering the last two dimensions as matrices
        advantage = q_soa - torch.einsum("...ii->...i", pi @ q_so.transpose(-2, -1))
        return advantage

    def compute_advantage(self, obs: torch.Tensor, agent_action: int, opponent_action: int | None) -> torch.Tensor:
        """Compute the TD delta (a.k.a. advantage). May consider intrinsic reward if an intrisic critic was provided."""
        advantage = self._compute_advantage_with_critic(obs, agent_action, opponent_action, self._critic)
        
        if self._intrinsic_critic is not None:
            intrinsic_advantage = self._compute_advantage_with_critic(obs, agent_action, opponent_action, self._intrinsic_critic)
            advantage += intrinsic_advantage
        
        return advantage
    
    def _learn_complete(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, intrinsic_reward: float, terminated: bool, truncated: bool, obs_agent_action: int, obs_opponent_action: int, next_obs_agent_action: int, next_obs_opponent_action: int):
        """Perform a complete update. This method is not called by the training loop since in practice it needs to be performed one-step later as we need to know the agent's and opponent's actual action on the next observation `next_obs`."""
        # Update the Q-table. Save the TD error in case the caller wants to check it. The TD error is None if no critic update was performed
        self.td_error = None
        if obs_agent_action is not None:
            next_obs_action_probabilities = self.actor.probabilities(next_obs, next_opponent_action=None).detach().squeeze().T
        else:
            next_obs_action_probabilities = torch.ones(self.action_dim, self.opponent_action_dim).float() / (self.action_dim)
        self.frameskipped_critic_updates.append(
            (obs, reward, intrinsic_reward, next_obs, terminated, obs_agent_action, obs_opponent_action, next_obs_action_probabilities, next_obs_agent_action, next_obs_opponent_action)
        )
        if next_obs_agent_action is not None or terminated or truncated:
            # Perform the frameskipped updates assuming no discounting (that's pretty much what's special about frameskipped updates)
            previous_discount = self._critic.discount
            self._critic.discount = 1.0
            total_td_error = 0.0

            # We perform the updates in reverse order so that the future reward is propagated back to the oldest retained updates
            for frameskipped_update in reversed(self.frameskipped_critic_updates):
                obs_, reward_, intrinsic_reward_, next_obs_, terminated_, obs_agent_action_, obs_opponent_action_, next_obs_action_probabilities_, next_obs_agent_action_, next_obs_opponent_action_ = frameskipped_update
                
                if self._update_style != self.UpdateStyle.EXPECTED_SARSA:
                    next_obs_action_probabilities_ = None
                if self._update_style != self.UpdateStyle.SARSA:
                    next_obs_agent_action_ = None
                
                # The kwargs that are shared between the critic and the intrinsic critic
                critic_kwargs = {
                    "obs": obs_,
                    "next_obs": next_obs_,
                    "agent_action": obs_agent_action_,
                    "opponent_action": obs_opponent_action_,
                    "next_agent_action": next_obs_agent_action_,
                    "next_opponent_action": None, # We ignore what the opponent actually did, so that we would not need importance sampling. This will assume that the opponent behaves uniformly randomly
                    "next_agent_policy": next_obs_action_probabilities_,
                    "next_opponent_policy": self._custom_opponent_policy(obs_).detach().T if self._custom_opponent_policy is not None else None,
                }

                td_error = self._critic.update(
                    reward=reward_,
                    terminated=terminated_,
                    **critic_kwargs,
                )

                # The intrinsic critic not only updates using the intrinsic reward only, but also doesn't consider episode boundaries (is non-episodic), as recommended by the RND paper
                if self._intrinsic_critic is not None:
                    td_error += self._intrinsic_critic.update(
                        reward=intrinsic_reward_,
                        terminated=False,
                        **critic_kwargs,
                    )

                total_td_error = total_td_error + torch.mean(td_error).item()
            
            self.td_error = total_td_error
            self._critic.discount = previous_discount
            self.frameskipped_critic_updates.clear()

        # If the agent action is None then that means the agent couldn't act, so it doesn't make sense to update the actor
        if obs_agent_action is None:
            pass
        # If the agent did perform an action, then update the actor
        else:
            # If the opponent is being frameskipped, then we consider them as doing any of their actions.
            # Otherwise, we just update the portion of the policy concerned with the actually performed opponent action.
            self._update_actor(obs, obs_agent_action, obs_opponent_action)

        if self.policy_cumulative_discount:
            self.cumulative_discount *= self.critic.discount

        if terminated or truncated:
            self.cumulative_discount = 1.0

    def learn(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, *, obs_agent_action: int, obs_opponent_action: int, intrinsic_reward: float = 0.0, **kwargs):
        """
        Update the actor and critic in this environment step. Should be preceded by an environment interaction with `sample_action()`.
        
        NOTE: actual learning occurs one-step later, since we need to know the opponent's actual action on the next observation

        Extra parameters
        ----------------
        - `obs_agent_action`: which action the agent actually performed. Should match what was sampled, but can be `None` if the critic update should be done for all agent actions
        - `obs_opponent_action`: which action the opponent actually performed. May not match what was used at the time of sampling
        - `intrinsic_reward`: the intrinsic reward from the environment
        """
        if self.postponed_learn is not None:
            self._learn_complete(**self.postponed_learn, next_obs_agent_action=obs_agent_action, next_obs_opponent_action=obs_opponent_action)
            
            # If the episode terminated, just learn the last bit, don't even need to wait for the agent and opponent actions since they won't exist
            if terminated or truncated:
                self._learn_complete(obs=obs, next_obs=next_obs, reward=reward, intrinsic_reward=intrinsic_reward, terminated=terminated, truncated=truncated, obs_agent_action=obs_agent_action, obs_opponent_action=obs_opponent_action, next_obs_agent_action=None, next_obs_opponent_action=None)
                self.postponed_learn = None
                return
            
        self.postponed_learn = {
            "obs": obs,
            "next_obs": next_obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "obs_agent_action": obs_agent_action,
            "obs_opponent_action": obs_opponent_action,
            "intrinsic_reward": intrinsic_reward,
        }

    @property
    def actor(self):
        return self._actor
    
    @property
    def critic(self):
        return self._critic

    @property
    def intrinsic_critic(self):
        return self._intrinsic_critic

    @property
    def actor_learning_rate(self) -> float:
        return self.actor_optimizer.param_groups[0]["lr"]

    @actor_learning_rate.setter
    def actor_learning_rate(self, learning_rate: float):
        self.actor_optimizer.param_groups[0]["lr"] = learning_rate
    
    @property
    def critic_learning_rate(self) -> float:
        return self._critic.learning_rate
    
    @critic_learning_rate.setter
    def critic_learning_rate(self, learning_rate: float):
        self._critic.learning_rate = learning_rate

    def consider_opponent_policy(self, opponent_policy: Callable[[torch.Tensor], torch.Tensor]):
        """
        Consider a specific opponent policy when calculating the Q-values.
        With this, the Q-values will be calculated assuming the opponent behaves according to the given policy.
        
        Parameters
        ----------
        - `opponent_policy`: A callable that takes an environment observation (tensor) and outputs the probability distribution over opponent actions (of size `batch_size X action_size`). If `None`, will assume a uniform random opponent
        """
        self._custom_opponent_policy = opponent_policy
