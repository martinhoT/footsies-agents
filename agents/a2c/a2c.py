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


def epoched(learning_method: Callable[..., None]):
    def wrapper(self, *args, **kwargs):
        self.epochs
        self.timesteps
        self.minibatch_number


        learning_method(self, *args, **kwargs)
    return wrapper


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
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, obs: torch.Tensor):
        rep = self._representation(obs)

        logits = self.actor_layers(rep)
        return self.softmax(logits)
    
    def from_representation(self, rep: torch.Tensor) -> torch.Tensor:
        logits = self.actor_layers(rep)
        return self.softmax(logits)
    
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
        action = action_distribution.sample()
        return action

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
        # Considers the action that the player performed in the next observation
        SARSA = 0
        # Considers the action distribution of the player in the next observation
        EXPECTED_SARSA = 1
        # Considers the player to act greedily in the next observation
        Q_LEARNING = 2
        # Considers the player to be acting uniformly randomly in the next observation
        UNIFORM = 3
    
    def __init__(
        self,
        actor: ActorNetwork,
        critic: QFunction,
        actor_entropy_loss_coef: float = 0.0,
        actor_learning_rate: float = 1e-4,
        policy_cumulative_discount: bool = False,
        actor_gradient_clipping: float = None,
        agent_update_style: UpdateStyle = UpdateStyle.EXPECTED_SARSA,
        opponent_update_style: UpdateStyle = UpdateStyle.EXPECTED_SARSA,
        intrinsic_critic: QFunction = None,
        alternative_advantage: bool = True,
        accumulate_at_frameskip: bool = False,
        broadcast_at_frameskip: bool = False,
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
        - `actor_gradient_clipping`: the maximum norm of the actor's gradients. If `None`, then no clipping is performed
        - `agent_update_style`: how to update the critic's Q-values considering the agent's policy
        - `opponent_update_style`: how to update the critic's Q-values considering the opponent's policy
        - `intrinsic_critic`: the critic for the intrinsic reward. If `None`, then no intrinsic reward is considered
        - `alternative_advantage`: whether to use the alternative advantage formula, which considers the reward as-is and is less reliant on the critic converging to the correct values.
        - `accumulate_at_frameskip`: whether to accumulate the updates when frameskipping. This way, the agent effectively.
        The environment will lose some theoretical properties, such as the Markov property and stationarity.
        It should, however, greatly facilitate credit assignment.
        This should allow the actor to learn sooner
        - `broadcast_at_frameskip`: when frameskipping (i.e. the agent's or opponent's action is `None`), we can do one of two things:
            - Either consider the agent/opponent to be doing any action (update on all actions), corresponding to `True`
            - Or consider the agent/opponent to still be doing the originating action before frameskipping (update on a single action), corresponding to `False`
        
        If using a Q-value table, it's recommended to use `broadcast_at_frameskip` with `True`, since it's more correct.
        However, if using a function approximator such as a neural network, it's likely that the broadcasts will leak into the state in which the original action
        was performed (since states are mostly similar), which will cause the Q-values to be uniform at that state.
        In such cases, it's recommended to use `broadcast_at_frameskip` with `False`.
        """
        if not isinstance(agent_update_style, self.UpdateStyle):
            raise ValueError(f"'agent_update_style' must be an instance of {self.UpdateStyle}, not {type(agent_update_style)}")
        if not isinstance(opponent_update_style, self.UpdateStyle):
            raise ValueError(f"'opponent_update_style' must be an instance of {self.UpdateStyle}, not {type(opponent_update_style)}")
        if agent_update_style == self.UpdateStyle.UNIFORM:
            LOGGER.warning("Considering the agent to be following a uniform random policy doesn't make much sense, and is not recommended")
        if agent_update_style == self.UpdateStyle.SARSA:
            raise NotImplementedError("SARSA updates are not supported for the agent")
        if opponent_update_style == self.UpdateStyle.SARSA:
            raise NotImplementedError("SARSA updates are not supported for the opponent")

        self._actor = actor
        self._critic = critic
        self._intrinsic_critic = intrinsic_critic
        self.actor_entropy_loss_coef = actor_entropy_loss_coef
        self.policy_cumulative_discount = policy_cumulative_discount
        self.consider_opponent_action = actor.consider_opponent_action
        self._actor_gradient_clipping = actor_gradient_clipping
        self._agent_update_style = agent_update_style
        self._opponent_update_style = opponent_update_style
        self._alternative_advantage = alternative_advantage
        self._accumulate_at_frameskip = accumulate_at_frameskip
        self._broadcast_at_frameskip = broadcast_at_frameskip

        self.action_dim = self._critic.action_dim
        self.opponent_action_dim = self._critic.opponent_action_dim

        # Due to the way the gradients are set up, we want the optimizer to maximize (i.e., leave the gradients' sign unchanged)
        self.actor_optimizer = torch.optim.SGD(self._actor.parameters(), lr=actor_learning_rate, maximize=True)

        self.frameskipped_critic_updates = []

        # If we are not performing broadcasting during frameskipping, we need to remember the last action performed
        # by the agent/opponent, the one that led to the frameskipping.
        # The last agent's action is already tracking in self.current_action.
        self._last_valid_agent_action = None
        self._last_valid_opponent_action = None

        # Discount throughout a single episode
        self.cumulative_discount = 1.0
        
        # Track values
        self.delta = 0.0
        self.td_error = 0.0

    def sample_action(self, obs: torch.Tensor, *, next_opponent_action: int, **kwargs) -> int:
        """Sample an action from the actor. A training step starts with `sample_action()`, followed immediately by an environment step and `learn()`."""    
        return self.actor.sample_action(obs, next_opponent_action=next_opponent_action).item()

    def _update_actor(self, obs: torch.Tensor, opponent_action: int | None, agent_action: int, delta: torch.Tensor):
        # Save the delta for tracking
        self.delta = delta.mean().item() # We might get a delta vector
        
        # Calculate the probability distribution at obs considering opponent action, and consider we did the given action
        action_probabilities = self.actor.probabilities(obs, opponent_action)
        action_log_probabilities = torch.log(action_probabilities + 1e-8)
        
        # Update the actor network
        self.actor_optimizer.zero_grad()

        actor_entropy = -torch.sum(action_log_probabilities * action_probabilities, dim=-1)
        actor_delta = self.cumulative_discount * delta * action_log_probabilities[..., agent_action]
        actor_score = torch.mean(actor_delta + self.actor_entropy_loss_coef * actor_entropy)
        actor_score.backward()

        if self._actor_gradient_clipping is not None:
            nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), self._actor_gradient_clipping)

        self.actor_optimizer.step()

        if LOGGER.isEnabledFor(logging.DEBUG):
            new_action_probabilities = self.actor.probabilities(obs, opponent_action).detach()
            new_entropy = -torch.sum(torch.log(new_action_probabilities + 1e-8) * new_action_probabilities, dim=-1)
            LOGGER.debug("Actor was updated using delta %s for action %s and opponent action %s, going from a distribution of %s with entropy %s to a distribution of %s with entropy %s", delta, agent_action, opponent_action, action_probabilities, actor_entropy, new_action_probabilities, new_entropy)

    def value(self, obs: torch.Tensor, opponent_action: int | None, critic: QFunction | None = None) -> torch.Tensor:
        """
        Get the value of a given state, according to the effective policy (depending on the set `update_style`).
        The returned tensor is a leaf tensor.
        
        The value is computed as a function of the Q-function and agent policy: `V(s, o) = pi(. | s, o).T Q(s, o, .)`.

        If `critic` is `None`, then both the extrinsic and intrinsic reward critics will be used.
        """
        if critic is None:
            extrinsic_value = self.value(obs, opponent_action, critic=self.critic)
            if self.intrinsic_critic is not None:
                intrinsic_value = self.value(obs, opponent_action, critic=self.intrinsic_critic)
            else:
                intrinsic_value = 0.0
            return extrinsic_value + intrinsic_value
        
        q_so = critic.q(obs, opponent_action=opponent_action).detach()

        if self._agent_update_style == self.UpdateStyle.EXPECTED_SARSA:
            pi = self.actor.probabilities(obs, opponent_action).detach()
        elif self._agent_update_style == self.UpdateStyle.Q_LEARNING:
            greedy_action = torch.argmax(q_so, dim=-1)
            pi = nn.functional.one_hot(greedy_action, num_classes=self.action_dim).float()
        elif self._agent_update_style == self.UpdateStyle.UNIFORM:
            n_rows = 1 if opponent_action is not None else self.opponent_action_dim
            pi = torch.ones(n_rows, self.action_dim).float() / self.action_dim

        # The policy doesn't need to be transposed since the last dimension is the action dimension.
        # This Einstein summation equation computes the diagonal of a >2D tensor by considering the last two dimensions as matrices.
        # Also, by design, the policy is already transposed (opp_actions X agent_actions).
        return torch.einsum("...ii->...i", pi @ q_so.transpose(-2, -1))

    def advantage(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, agent_action: int, opponent_action: int | None, next_obs_opponent_policy: torch.Tensor | None, critic: QFunction = None) -> torch.Tensor:
        """
        Compute the advantage. May consider intrinsic reward if an intrinsic critic was provided.
        The returned tensor is a leaf tensor.
        
        The value is computed as a function of both the Q-function and agent policy:
        - original: `A(s, o, a) = Q(s, o, a) - V(s, o) = Q(s, o, a) - pi(. | s, o).T Q(s, o, .)`
        - alternative: `... = R + pi(. | s', o).T Q(s', o, .) - pi(. | s, o).T Q(s, o, .)`
        """
        if self._alternative_advantage:
            if terminated:
                v_so_next = 0.0
            
            else:
                v_s_next = self.value(next_obs, opponent_action=None, critic=critic).detach().squeeze(0)
                
                if self._opponent_update_style == self.UpdateStyle.EXPECTED_SARSA:
                    next_opp_pi = next_obs_opponent_policy
                elif self._opponent_update_style == self.UpdateStyle.Q_LEARNING:
                    next_opp_pi = nn.functional.one_hot(torch.argmin(v_s_next), num_classes=self.opponent_action_dim).unsqueeze(1).float()
                elif self._opponent_update_style == self.UpdateStyle.UNIFORM:
                    next_opp_pi = torch.ones(self.opponent_action_dim, 1).float() / self.opponent_action_dim
                
                v_so_next = next_opp_pi.T @ v_s_next

            q_soa = reward + self.critic.discount * v_so_next

        else:
            q_soa = critic.q(obs, agent_action, opponent_action).detach()
        
        v_so = self.value(obs, opponent_action, critic=critic).detach()

        advantage = q_soa - v_so
        return advantage
    
    def learn(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        reward: float,
        terminated: bool,
        truncated: bool,
        *,
        obs_agent_action: int | None,
        obs_opponent_action: int | None,
        agent_will_frameskip: bool,
        opponent_will_frameskip: bool,
        next_obs_opponent_policy: torch.Tensor | None = None,
        intrinsic_reward: float = 0.0,
        **kwargs,
    ):
        """
        Update the actor and critic in this environment step.
        
        Extra parameters
        ----------------
        - `obs_agent_action`: which action the agent actually performed. Should match what was sampled, but can be `None` if the critic update should consider the agent to be unactionable
        - `obs_opponent_action`: which action the opponent actually performed. May not match what was used at the time of sampling. Can be `None` if the opponent is unactionable
        - `agent_will_frameskip`: whether the agent will be unactionable in the next step
        - `opponent_will_frameskip`: whether the opponent will be unactionable in the next step
        - `next_obs_opponent_policy`: the probability distribution over opponent actions for the next observation. Should be a column vector. Can be `None` as long as the update style is not Expected SARSA, or the episode terminated
        - `intrinsic_reward`: the intrinsic reward from the environment
        """
        # Update the Q-table. Save the TD error in case the caller wants to check it. The TD error is None if no critic update was performed
        self.td_error = None

        # Update the last valid action that the agent/opponent performed, i.e. out of any frameskipping.
        # This is needed if we don't broadcast when frameskipping, since we need to know which action the opponent did perform.
        if obs_agent_action is not None:
            self._last_valid_agent_action = obs_agent_action
        elif self._last_valid_agent_action is None:
            raise RuntimeError("the agent is being frameskipped, but we don't remember them having ever performed an action")
        
        if obs_opponent_action is not None:
            self._last_valid_opponent_action = obs_opponent_action
        elif self._last_valid_opponent_action is None:
            raise RuntimeError("the opponent is being frameskipped, but we don't remember them having ever performed an action")

        if not self.broadcast_at_frameskip:
            obs_agent_action = self._last_valid_agent_action

        if not self.broadcast_at_frameskip:
            obs_opponent_action = self._last_valid_opponent_action

        # If the agent will be frameskipped...
        if agent_will_frameskip:
            # ... then we consider they will be doing any of their actions (uniform random policy).
            if self._broadcast_at_frameskip:
                next_obs_agent_policy = torch.ones(self.action_dim, self.opponent_action_dim).float() / (self.action_dim)
            # ... then we consider they will keep doing the action before frameskipping.
            else:
                next_obs_agent_policy = nn.functional.one_hot(torch.tensor(obs_agent_action), num_classes=self.action_dim).float().unsqueeze(1).expand(-1, self.opponent_action_dim)
        
        # If the agent could act, then it makes sense for it to have a policy.
        # We have to construct it manually since it is not provided as an argument.
        # It doesn't make sense to provide the policy as argument, since we already have access to it here.
        # On the other hand, we don't have access to the opponent model, hence why it is explicitly provided.
        else:
            # Use the agent's next policy
            if self._agent_update_style == self.UpdateStyle.EXPECTED_SARSA:
                # We don't consider the action that the opponent will perform, this is just the Q-value matrix to provide to the critic method,
                # which will then sort how to consider the opponent's next actions internally.
                next_obs_agent_policy = self.actor.probabilities(next_obs, next_opponent_action=None).detach().squeeze().T
                pass
            # Enable Q-learning, change agent's policy to greedy
            elif self._agent_update_style == self.UpdateStyle.Q_LEARNING:
                next_obs_agent_policy = "greedy"
            # Change agent's policy to be uniform
            elif self._agent_update_style == self.UpdateStyle.UNIFORM:
                next_obs_agent_policy = "uniform"
        
        # If the opponent will be frameskipped...
        if opponent_will_frameskip:
            # ... then we consider they will be doing any of their actions (uniform random policy).
            if self._broadcast_at_frameskip:
                next_obs_opponent_policy = torch.ones(self.opponent_action_dim, 1).float() / (self.opponent_action_dim)
            # ... then we consider they will keep doing the action before frameskipping.
            else:
                next_obs_opponent_policy = nn.functional.one_hot(torch.tensor(obs_opponent_action), num_classes=self.opponent_action_dim).float().unsqueeze(1)

        else:
            # Keep the opponent's next policy, which should have been provided as an argument
            if self._opponent_update_style == self.UpdateStyle.EXPECTED_SARSA:
                if next_obs_opponent_policy is None:
                    raise ValueError("Expected SARSA updates for the opponent require the opponent's policy on the next observation to be provided")
            # Enable Q-learning, change opponent's policy to greedy
            elif self._opponent_update_style == self.UpdateStyle.Q_LEARNING:
                next_obs_opponent_policy = "greedy"
            # Change opponent's policy to be uniform
            elif self._opponent_update_style == self.UpdateStyle.UNIFORM:
                next_obs_opponent_policy = "uniform"

        # Schedule a critic update
        self.frameskipped_critic_updates.append(
            (obs, reward, next_obs, terminated, obs_agent_action, obs_opponent_action, next_obs_agent_policy, next_obs_opponent_policy, intrinsic_reward)
        )

        # If the agent will not be frameskipped, then we can perform the updates now.
        # If the game is finished, then we also perform all scheduled updates.
        if (not agent_will_frameskip) or terminated or truncated:
            total_reward = sum(update[1] for update in self.frameskipped_critic_updates)
            total_intrinsic_reward = sum(update[8] for update in self.frameskipped_critic_updates)

            if self._accumulate_at_frameskip:
                obs_ = self.frameskipped_critic_updates[0][0]
                next_obs_ = self.frameskipped_critic_updates[-1][2]
                obs_agent_action_ = self.frameskipped_critic_updates[0][4]
                obs_opponent_action_ = self.frameskipped_critic_updates[0][5] # we assume the first opponent action, as it's technically the last one we see
                next_obs_agent_policy_ = self.frameskipped_critic_updates[-1][6]
                next_obs_opponent_policy_ = self.frameskipped_critic_updates[-1][7]

                # The kwargs that are shared between the critic and the intrinsic critic
                critic_kwargs = {
                    "obs": obs_,
                    "next_obs": next_obs_,
                    "agent_action": obs_agent_action_,
                    "opponent_action": obs_opponent_action_,
                    "next_agent_policy": next_obs_agent_policy_,
                    "next_opponent_policy": next_obs_opponent_policy_,
                }

                td_error = self._critic.update(
                    reward=total_reward,
                    terminated=terminated,
                    **critic_kwargs,
                )

                # The intrinsic critic not only updates using the intrinsic reward only, but also doesn't consider episode boundaries (is non-episodic), as recommended by the RND paper
                if self._intrinsic_critic is not None:
                    td_error += self._intrinsic_critic.update(
                        reward=total_intrinsic_reward,
                        terminated=False,
                        **critic_kwargs,
                    )
                
                td_error = torch.mean(td_error).item()

            else:
                # Perform the frameskipped updates assuming no discounting (that's pretty much what's special about frameskipped updates)
                previous_discount = self._critic.discount
                self._critic.discount = 1.0

                # We perform the updates in reverse order so that the future reward is propagated back to the oldest retained updates
                for frameskipped_update in reversed(self.frameskipped_critic_updates):
                    obs_, reward_, next_obs_, terminated_, obs_agent_action_, obs_opponent_action_, next_obs_agent_policy_, next_obs_opponent_policy_, intrinsic_reward_ = frameskipped_update

                    # The kwargs that are shared between the critic and the intrinsic critic
                    critic_kwargs = {
                        "obs": obs_,
                        "next_obs": next_obs_,
                        "agent_action": obs_agent_action_,
                        "opponent_action": obs_opponent_action_,
                        "next_agent_policy": next_obs_agent_policy_,
                        "next_opponent_policy": next_obs_opponent_policy_,
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

                    td_error = torch.mean(td_error).item()
                    self._critic.discount = previous_discount

            # The agent only performed an action at the beginning of frameskipping. It doesn't make sense to update at any other observation.
            obs_ = self.frameskipped_critic_updates[0][0]
            next_obs_ = self.frameskipped_critic_updates[-1][2]
            obs_agent_action_ = self.frameskipped_critic_updates[0][4]
            obs_opponent_action_ = self.frameskipped_critic_updates[0][5]
            next_obs_opponent_policy_ = self.frameskipped_critic_updates[-1][7]
            extrinsic_advantage = self.advantage(obs_, next_obs_, total_reward, terminated, obs_agent_action_, obs_opponent_action_, next_obs_opponent_policy_, critic=self.critic)
            if self.intrinsic_critic is not None:
                intrinsic_advantage = self.advantage(obs_, next_obs_, total_intrinsic_reward, False, obs_agent_action_, obs_opponent_action_, next_obs_opponent_policy_, critic=self.intrinsic_critic)
            else:
                intrinsic_advantage = 0.0

            self._update_actor(obs_, obs_opponent_action_, obs_agent_action_, extrinsic_advantage + intrinsic_advantage)

            self.td_error = td_error
            self.frameskipped_critic_updates.clear()

        if self.policy_cumulative_discount:
            self.cumulative_discount *= self.critic.discount

        if terminated or truncated:
            self.cumulative_discount = 1.0
            self._last_valid_agent_action = None
            self._last_valid_opponent_action = None
        
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
    
    @property
    def broadcast_at_frameskip(self) -> bool:
        return self._broadcast_at_frameskip
    
    @broadcast_at_frameskip.setter
    def broadcast_at_frameskip(self, value: bool):
        self._broadcast_at_frameskip = value
