import torch
from torch import nn
from torch.distributions import Categorical
from agents.torch_utils import create_layered_network
from agents.utils import extract_sub_kwargs
from agents.ql.ql import QFunction
from abc import ABC, abstractmethod


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


class A2CLearnerBase(ABC):
    @abstractmethod
    def sample_action(self, obs: torch.Tensor, **kwargs) -> int:
        pass

    @abstractmethod
    def learn(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, **kwargs):
        pass


class A2CLambdaLearner(A2CLearnerBase):
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
        policy_improvement_steps: int = 1,
        policy_evaluation_steps: int = 1,
        policy_cumulative_discount: bool = True,
        **kwargs,
    ):
        """Implementation of the advantage actor-critic algorithm with eligibility traces, from the Sutton & Barto book"""
        self.actor = actor
        self.critic = critic
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

        actor_optimizer_kwargs, critic_optimizer_kwargs = extract_sub_kwargs(kwargs, ("actor_optimizer", "critic_optimizer"))

        actor_optimizer_kwargs = {
            "lr": 1e-4,
            **actor_optimizer_kwargs,
        }
        critic_optimizer_kwargs = {
            "lr": 1e-4,
            **critic_optimizer_kwargs,
        }

        # Due to the way the gradients are set up, we want the optimizer to maximize (i.e., leave the gradients' sign unchanged)
        self.actor_optimizer = actor_optimizer(self.actor.parameters(), maximize=True, **actor_optimizer_kwargs)
        self.critic_optimizer = critic_optimizer(self.critic.parameters(), maximize=True, **critic_optimizer_kwargs)

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
        action_probabilities = self.actor(obs)
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

        critic_score = self.critic(obs)
        critic_score.backward()
        with torch.no_grad():
            for critic_trace, parameter in zip(self.critic_traces, self.critic.parameters()):
                critic_trace.copy_(self.discount * self.critic_lambda * critic_trace + parameter.grad)
                parameter.grad.copy_(self.delta * critic_trace)

        self.critic_optimizer.step()

    def _update_actor(self):
        self.actor_optimizer.zero_grad()

        actor_score = self.cumulative_discount * self.action_distribution.log_prob(self.action)
        actor_score.backward(retain_graph=True)
        with torch.no_grad():
            for actor_trace, parameter in zip(self.actor_traces, self.actor.parameters()):
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
                target = reward + self.discount * self.critic(next_obs)
            
            delta = (target - self.critic(obs)).item()
    
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


# NOTE: we can use both players' perspectives when updating the Q-value function. This is left to the training loop to manage
class A2CQLearner(A2CLearnerBase):
    def __init__(
        self,
        actor: ActorNetwork,
        critic: QFunction,
        actor_entropy_loss_coef: float = 0.0,
        actor_optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        policy_cumulative_discount: bool = True,
        consider_opponent_action: bool = False,
        **kwargs,
    ):
        """Implementation of a custom actor-critic algorithm with a Q-value table for the critic"""
        self.actor = actor
        self.critic = critic
        self.discount = critic.discount
        self.actor_entropy_loss_coef = actor_entropy_loss_coef
        self.policy_cumulative_discount = policy_cumulative_discount
        self.consider_opponent_action = consider_opponent_action

        actor_optimizer_kwargs, = extract_sub_kwargs(kwargs, ("actor_optimizer",))

        actor_optimizer_kwargs = {
            "lr": 1e-4,
            **actor_optimizer_kwargs,
        }

        # Due to the way the gradients are set up, we want the optimizer to maximize (i.e., leave the gradients' sign unchanged)
        self.actor_optimizer = actor_optimizer(self.actor.parameters(), maximize=True, **actor_optimizer_kwargs)

        self.action_distribution = None
        self.action = None
        self.postponed_learn: dict = None

        # Discount throughout a single episode
        self.cumulative_discount = 1.0
        
        # Track values
        self.delta = 0.0
        self.td_error = 0.0

    def compute_action_distribution(self, obs: torch.Tensor, next_opponent_action: int) -> torch.distributions.Distribution:
        """Get the action probability distribution for the given observation and predicted opponent action."""
        obs = self._append_opponent_action_if_needed(obs, next_opponent_action)
        action_probabilities = self.actor(obs)
        return Categorical(probs=action_probabilities)

    def sample_action(self, obs: torch.Tensor, *, next_opponent_action: int, **kwargs) -> int:
        """Sample an action from the actor. A training step starts with `sample_action()`, followed immediately by an environment step and `learn()`."""    
        self.action_distribution = self.compute_action_distribution(obs, next_opponent_action)
        self.action = self.action_distribution.sample()
        return self.action.item()

    def _append_opponent_action_if_needed(self, obs: torch.Tensor, opponent_action: int) -> torch.Tensor:
        if not self.consider_opponent_action:
            return obs
        
        if self.consider_opponent_action and opponent_action is None:
                raise ValueError("a prediction for the opponent's next action must be provided when choosing how to act, since we are considering opponent actions")

        opponent_action_onehot = nn.functional.one_hot(torch.tensor([opponent_action]), num_classes=self.critic.opponent_action_dim)
        return torch.hstack((obs, opponent_action_onehot))

    def _update_actor(self, obs: torch.Tensor, agent_action: int, opponent_action: int):
        # Compute the TD delta
        delta = self.compute_advantage(obs, agent_action, opponent_action)
        self.delta = delta
        
        # Calculate the probability distribution at obs considering opponent action, and consider we did the given action
        action_distribution = self.compute_action_distribution(obs, opponent_action)
        action_tensor = torch.tensor(agent_action, dtype=torch.int64)
        
        # Update the actor network
        self.actor_optimizer.zero_grad()

        actor_score = (1 - self.actor_entropy_loss_coef) * self.cumulative_discount * delta * action_distribution.log_prob(action_tensor) + self.actor_entropy_loss_coef * action_distribution.entropy()
        actor_score.backward()

        self.actor_optimizer.step()

    def compute_advantage(self, obs: torch.Tensor, agent_action: int, opponent_action: int) -> float:
        """Compute the TD delta (a.k.a. advantage)."""
        # If the opponent's action doesn't matter, since it's ineffectual, then compute the advantage considering all possible opponent actions.
        # NOTE: another possibility would be to perform random sampling? But then it would probably take a long time before convergence...
        if opponent_action is None:
            obs_with_opp = torch.vstack(self._append_opponent_action_if_needed(obs, o) for o in range(self.critic.opponent_action_dim))
        else:
            obs_with_opp = self._append_opponent_action_if_needed(obs, opponent_action)

        # NOTE: since the tensors are always on CPU, we don't need to call .numpy() with force=True
        # A(s, o, a) = Q(s, o, a) - V(s, o) = Q(s, o, a) - pi.T Q(s, o, .)
        q_soa = self.critic.q(obs.numpy().squeeze(), agent_action, opponent_action)
        pi = self.actor(obs_with_opp).detach().numpy()
        q_so = self.critic.q(obs.numpy().squeeze(), opponent_action=opponent_action)
        return q_soa - (pi @ q_so).item()
    
    def _learn_complete(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, obs_agent_action: int, obs_opponent_action: int, next_obs_opponent_action: int):
        """Perform a complete update. This method is not called by the training loop since in practice it needs to be performed one-step later as we need to know the opponent's actual action on the next observation `next_obs`."""
        # Update the Q-table. Save the TD error in case the caller wants to check it
        self.td_error = self.critic.update(obs.numpy().squeeze(), obs_agent_action, reward, next_obs.numpy().squeeze(), terminated, obs_opponent_action, next_obs_opponent_action)

        # If the agent action is None then that means the agent couldn't act, so it doesn't make sense to update the actor
        if obs_agent_action is None:
            # TODO: save in a buffer for later update
            # self.agent_action_buffer.append(obs, next_obs, reward)
            
            pass
        # If the agent did perform an action, then update the actor
        else:
            # If the opponent is being frameskipped, then we consider them as doing any of their actions, and update accordingly
            if obs_opponent_action is None:
                # TODO: how about considering it to be 0? (i.e. STAND, the action of doing nothing)
                for opponent_action in range(self.critic.opponent_action_dim):
                    self._update_actor(obs, obs_agent_action, opponent_action)
            # We have to recalculate the policy's action distribution since the opponent action is likely different
            else:
                self._update_actor(obs, obs_agent_action, obs_opponent_action)

        if self.policy_cumulative_discount:
            self.cumulative_discount *= self.discount

        if terminated or truncated:
            self.cumulative_discount = 1.0

    # TODO: can we use importance sampling weights to make the update of the Q-value more efficient by not explicitly considering just one action from the opponent but all of them? Of course, we need access to the opponent's policy, which we do during training supposedly
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
            self._learn_complete(**self.postponed_learn, next_obs_opponent_action=obs_opponent_action)
            
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
