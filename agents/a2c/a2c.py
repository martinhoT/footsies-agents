import torch as T
import torch.nn.functional as F
import logging
from torch import nn
from torch.distributions import Categorical
from agents.torch_utils import create_layered_network, ToMatrix
from agents.ql.ql import QFunction
from abc import ABC, abstractmethod
from enum import Enum
from agents.action import ActionMap
from collections import deque
from dataclasses import dataclass

LOGGER = logging.getLogger("main.a2c")


class ValueNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_layer_sizes: list[int] | None = None,
        hidden_layer_activation: type[nn.Module] = nn.Identity,
        representation: nn.Module | None = None,
    ):
        super().__init__()

        self.critic_layers = create_layered_network(obs_dim, 1, hidden_layer_sizes, hidden_layer_activation)
        self.representation = nn.Identity() if representation is None else representation

    def forward(self, obs: T.Tensor):
        rep = self.representation(obs)

        return self.critic_layers(rep)

    def from_representation(self, rep: T.Tensor) -> T.Tensor:
        return self.critic_layers(rep)


class ActorNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_layer_sizes: list[int] | None = None,
        hidden_layer_activation: type[nn.Module] = nn.Identity,
        representation: nn.Module | None = None,
        opponent_action_dim: int | None = None,
        footsies_masking: bool = True,
        p1: bool = True,
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
        self._footsies_masking = footsies_masking
        self._p1 = p1

        consider_opponent_action = opponent_action_dim is not None
        
        self._consider_opponent_action = consider_opponent_action

        output_dim = action_dim * (opponent_action_dim if consider_opponent_action else 0)
        self.actor_layers = create_layered_network(obs_dim, output_dim, hidden_layer_sizes, hidden_layer_activation)
        if consider_opponent_action:
            self.actor_layers.append(ToMatrix(opponent_action_dim, action_dim))
        self.softmax = nn.Softmax(dim=-1)

        # What actions can be performed during hitstop
        self._hitstop_mask = T.zeros((1, opponent_action_dim, action_dim), dtype=T.bool)
        for simple in ActionMap.PERFORMABLE_SIMPLES_IN_HITSTOP_INT:
            self._hitstop_mask[..., simple] = True

    def forward(self, obs: T.Tensor, temperature: float = 1.0):
        if self._footsies_masking:
            in_hitstop = ActionMap.is_in_hitstop_torch(obs, p1=self._p1)
            action_mask = T.ones_like(self._hitstop_mask).repeat(obs.size(0), 1, 1)
            action_mask[in_hitstop] = self._hitstop_mask
 
        else:
            action_mask = None
        
        rep = self._representation(obs)
        return self.from_representation(rep, action_mask, temperature=temperature)
    
    def from_representation(self, rep: T.Tensor, action_mask: T.Tensor | None = None, temperature: float = 1.0) -> T.Tensor:
        logits = self.actor_layers(rep)
        if action_mask is not None:
            # Invalidate all actions that are not in the mask
            logits = logits.masked_fill(~action_mask, -T.inf)
        return self.softmax(logits / temperature)
    
    @property
    def consider_opponent_action(self) -> bool:
        return self._consider_opponent_action
    
    def probabilities(self, obs: T.Tensor, next_opponent_action: T.Tensor | int | None, temperature: float = 1.0) -> T.Tensor:
        """Get the action probability distribution for the given observation and predicted opponent action. If the next opponent action is a tensor, then it should be 1-dimensional."""
        if next_opponent_action is None:
            next_opponent_action_idx = slice(None)
        else:
            next_opponent_action_idx = next_opponent_action

        if isinstance(next_opponent_action_idx, (int, slice)):
            action_probabilities = self(obs, temperature=temperature)[:, next_opponent_action_idx, :]
            return action_probabilities

        action_probabilities: T.Tensor = self(obs, temperature=temperature)
        return action_probabilities.take_along_dim(next_opponent_action_idx[:, None, None], dim=1)

    def distribution_size(self, obs: T.Tensor) -> int:
        """The effective size of the probability distribution when the agent acts at the given observation."""
        in_hitstop = ActionMap.is_in_hitstop_torch(obs, p1=self._p1)
        return 2 if in_hitstop else self._action_dim

    def decision_distribution(self, obs: T.Tensor, next_opponent_policy: T.Tensor, detached: bool = False, temperature: float = 1.0) -> Categorical:
        """Get the decision probabilities and distribution for the given observation and next opponent policy. The next opponent policy should have dimensions `batch_dim X opponent_action_dim`."""
        actor_probs = self.probabilities(obs, None, temperature=temperature)
        if detached:
            actor_probs = actor_probs.detach()
        probs = next_opponent_policy @ actor_probs
        # Squeeze the inner 1 dimension which is a consequence of 2D @ 3D tensor multiplication.
        # Example: 1x9 @ {1}x9x7 is equivalent to {1} matrix multiplication of 1x9 @ 9x7 -> 1x7, and so the output is {1}x1x7.
        probs = probs.squeeze(1)
        distribution = Categorical(probs=probs)
        return distribution

    def sample_action(self, obs: T.Tensor, next_opponent_action: int) -> int:
        """Randomly sample an action. This should be essentially the same as sampling from the `decision_distribution`, if `next_opponent_action` was sampled from `next_opponent_policy`."""
        action_probabilities = self.probabilities(obs, next_opponent_action)
        action_distribution = Categorical(probs=action_probabilities)
        action = action_distribution.sample()
        return int(action.item())

    def clone(self, p1: bool = True) -> "ActorNetwork":
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
            footsies_masking=self._footsies_masking,
            p1=p1,
        )

        cloned.load_state_dict(self.state_dict())

        return cloned


class A2CLearnerBase(ABC):
    @abstractmethod
    def sample_action(self, obs: T.Tensor, **kwargs) -> int:
        pass

    @abstractmethod
    def learn(self, obs: T.Tensor, next_obs: T.Tensor, reward: float, terminated: bool, truncated: bool, **kwargs):
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
            T.zeros_like(parameter)
            for parameter in actor.parameters()
        ]
        self.critic_traces = [
            T.zeros_like(parameter)
            for parameter in critic.parameters()
        ]

        # Due to the way the gradients are set up, we want the optimizer to maximize (i.e., leave the gradients' sign unchanged)
        self.actor_optimizer = T.optim.SGD(self._actor.parameters(), maximize=True, lr=actor_learning_rate) # type: ignore
        self.critic_optimizer = T.optim.SGD(self._critic.parameters(), maximize=True, lr=critic_learning_rate) # type: ignore

        self.action_distribution = None
        self.action = T.tensor(0)

        # Variables for policy improvement
        self.at_policy_improvement = True
        # Track at which environment step we are since we last performed policy improvement/evaluation
        self.policy_iteration_step = 0

        # Discount throughout a single episode
        self.cumulative_discount = 1.0
        
        # Track values
        self.delta = 0.0

    def sample_action(self, obs: T.Tensor, **kwargs) -> int:
        """Sample an action from the actor. A training step starts with `sample_action()`, followed immediately by an environment step and `learn()`."""    
        action_probabilities = self._actor(obs)
        self.action_distribution = Categorical(probs=action_probabilities)
        self.action = self.action_distribution.sample()
        
        return int(self.action.item())

    def _step_policy_iteration(self):
        step_threshold = self.policy_improvement_steps if self.at_policy_improvement else self.policy_evaluation_steps
        
        self.policy_iteration_step += 1
        if self.policy_iteration_step >= step_threshold:
            self.at_policy_improvement = not self.at_policy_improvement
            self.policy_iteration_step = 0

    def _update_critic(self, obs: T.Tensor):
        self.critic_optimizer.zero_grad()

        critic_score = self._critic(obs)
        critic_score.backward()
        with T.no_grad():
            for critic_trace, parameter in zip(self.critic_traces, self._critic.parameters()):
                critic_trace.copy_(self.discount * self.critic_lambda * critic_trace + parameter.grad)
                parameter.grad.copy_(self.delta * critic_trace) # type: ignore

        self.critic_optimizer.step()

    def _update_actor(self):
        if self.action_distribution is None:
            raise RuntimeError("attempted to update actor before an action has been sampled")

        self.actor_optimizer.zero_grad()

        actor_score = self.cumulative_discount * self.action_distribution.log_prob(self.action)
        actor_score.backward(retain_graph=True)
        with T.no_grad():
            for actor_trace, parameter in zip(self.actor_traces, self._actor.parameters()):
                actor_trace.copy_(self.discount * self.actor_lambda * actor_trace + parameter.grad)
                parameter.grad.copy_(self.delta * actor_trace * (1 - self.actor_entropy_loss_coef)) # type: ignore
        # Add the entropy score gradient
        entropy_score = self.actor_entropy_loss_coef * self.action_distribution.entropy()
        entropy_score.backward()

        self.actor_optimizer.step()

    def compute_advantage(self, obs: T.Tensor, next_obs: T.Tensor, reward: float, terminated: bool) -> float:
        """Compute the TD delta (a.k.a. advantage)."""
        with T.no_grad():
            if terminated:
                target = reward
            else:
                target = reward + self.discount * self._critic(next_obs)
            
            delta = (target - self._critic(obs)).item()
    
        return delta

    # TODO: if we use linear function approximation, could we do True Online TD(lambda)?
    def learn(self, obs: T.Tensor, next_obs: T.Tensor, reward: float, terminated: bool, truncated: bool, **kwargs):
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
        actor_gradient_clipping: float | None = None,
        agent_update_style: UpdateStyle = UpdateStyle.EXPECTED_SARSA,
        opponent_update_style: UpdateStyle = UpdateStyle.EXPECTED_SARSA,
        intrinsic_critic: QFunction | None = None,
        maxent: float = 0.0,
        maxent_gradient_flow: bool = False,
        ppo_objective: bool = False,
        ppo_objective_clip_coef: float = 0.2,
        alternative_advantage: bool = True,
        accumulate_at_frameskip: bool = False,
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
        - `maxent`: the coefficient for the entropy term of the MaxEnt RL objective, which encourages maximization of the policy's expected entropy. Doesn't make much sense to use this along with `actor_entropy_loss_coef`.
        It's recommended to use the alternative advantage formula in case this is used, since the entropy term will be able to always be considered in the actor update
        - `maxent_gradient_flow`: whether to let the gradient flow through the entropy term in the actor loss. This is what is commonly seen in implementations, but I can't find the reason why we do this.
        Doesn't do anything if the alternative advantage formula is used
        - `ppo_objective`: whether to use the clipped surrogate objective of the Proximal Policy Optimization algorithm, which considers the ratio of the new and old action probabilities when optimizing over multiple epochs
        - `ppo_objective_clip_coef`: the coefficient for the clipping of the ratio of the new and old action probabilities. If the ratio is outside of the range `[1 - clip_coef, 1 + clip_coef]`, then it is clipped to that range
        - `alternative_advantage`: whether to use the alternative advantage formula, which considers the reward as-is and is less reliant on the critic converging to the correct values.
        - `accumulate_at_frameskip`: whether to accumulate the updates when frameskipping.
        The environment will lose some theoretical properties, such as the Markov property and stationarity.
        It should, however, greatly facilitate credit assignment.
        This should allow the actor to learn sooner
        """
        if not isinstance(agent_update_style, self.UpdateStyle):
            raise ValueError(f"'agent_update_style' must be an instance of {self.UpdateStyle}, not {type(agent_update_style)}")
        if not isinstance(opponent_update_style, self.UpdateStyle):
            raise ValueError(f"'opponent_update_style' must be an instance of {self.UpdateStyle}, not {type(opponent_update_style)}")
        if agent_update_style == self.UpdateStyle.UNIFORM:
            LOGGER.warning("Considering the agent to be following a uniform random policy doesn't make much sense, and is not recommended")
        if agent_update_style == self.UpdateStyle.SARSA:
            raise NotImplementedError("SARSA updates are not supported for the agent")
        if maxent != 0.0 and actor_entropy_loss_coef != 0.0:
            LOGGER.warning("Using both the MaxEnt RL objective and the actor entropy loss coefficient is not recommended. If an effect similar to the entropy loss is meant to be used, activate the `maxent_gradient_flow` argument")
        if maxent != 0.0 and maxent_gradient_flow and not alternative_advantage:
            LOGGER.warning("Using gradient flow for the MaxEnt RL objective without the alternative advantage formula is inconsequential, and is probably not intended")

        self._actor = actor
        self._critic = critic
        self._intrinsic_critic = intrinsic_critic
        self.actor_entropy_loss_coef = actor_entropy_loss_coef
        self.policy_cumulative_discount = policy_cumulative_discount
        self.consider_opponent_action = actor.consider_opponent_action
        self._actor_gradient_clipping = actor_gradient_clipping
        self._agent_update_style = agent_update_style
        self._opponent_update_style = opponent_update_style
        self._maxent = maxent
        self._maxent_gradient_flow = maxent_gradient_flow
        self._ppo_objective = ppo_objective
        self._ppo_objective_clip_coef = ppo_objective_clip_coef
        self._alternative_advantage = alternative_advantage
        self._accumulate_at_frameskip = accumulate_at_frameskip

        self.action_dim = self._critic.action_dim
        self.opponent_action_dim = self._critic.opponent_action_dim

        # Due to the way the gradients are set up, we want the optimizer to maximize (i.e., leave the gradients' sign unchanged)
        self.actor_optimizer = T.optim.SGD(self._actor.parameters(), lr=actor_learning_rate, maximize=True) # type: ignore

        self.frameskipped_critic_updates: list[A2CQLearner.FrameskippedUpdate] = []

        # Needed for Sarsa-like updates
        self._delayed_updates: deque[A2CQLearner.LearnUpdate] = deque([], maxlen=2)

        # Discount throughout a single episode
        self.cumulative_discount = 1.0
        
        # Optional learning
        self._learn_actor = True
        self._learn_critic = True

        # Track values
        self.delta = 0.0
        self.extrinsic_td_error = 0.0
        self.intrinsic_td_error = 0.0

    def sample_action(self, obs: T.Tensor, *, next_opponent_action: int, **kwargs) -> int:
        """Sample an action from the actor. A training step starts with `sample_action()`, followed immediately by an environment step and `learn()`."""    
        return self.actor.sample_action(obs, next_opponent_action=next_opponent_action)

    def maxent_reward(self, policy: T.Tensor) -> T.Tensor:
        """Compute the entropy term of the MaxEnt RL objective according to the agent's policy."""
        return self._maxent * Categorical(probs=policy).entropy()

    # @epoched(timesteps=256, epochs=5, minibatch_size=32)
    # def _update_actor_ppo(self, obs: T.Tensor, opponent_action: T.Tensor | int, agent_action: T.Tensor | int, delta: T.Tensor, *, epoch_data: dict | None = None):
    #     action_probabilities = self.actor.probabilities(obs, opponent_action)
    #     action_log_probabilities = T.log(action_probabilities + 1e-8)
    #     action_log_probability = action_log_probabilities.take_along_dim(agent_action[:, None, None], dim=-1)

    #     # Update the actor network
    #     self.actor_optimizer.zero_grad()

    #     actor_entropy = -T.sum(action_log_probabilities * action_probabilities, dim=-1)

    #     if epoch_data is not None:
    #         old_action_log_probability = epoch_data.setdefault("action_log_probability", action_log_probability.detach())
    #     else:
    #         old_action_log_probability = action_log_probability.detach()

    #     ratio = (action_log_probability / old_action_log_probability).squeeze()
    #     actor_delta = T.min(self.cumulative_discount * delta * ratio, self.cumulative_discount * delta * T.clamp(ratio, 1 - self._ppo_objective_clip_coef, 1 + self._ppo_objective_clip_coef))

    #     actor_score = actor_delta.mean() + self.actor_entropy_loss_coef * actor_entropy.mean()
    #     actor_score.backward()

    #     if self._actor_gradient_clipping is not None:
    #         nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), self._actor_gradient_clipping)

    #     self.actor_optimizer.step()

    #     if LOGGER.isEnabledFor(logging.DEBUG):
    #         LOGGER.debug("Gradient step for the actor performed with score %s", actor_score.item())

    def _update_actor(self, obs: T.Tensor, opponent_action: int | None, agent_action: int, delta: T.Tensor):
        # Save the delta for tracking
        self.delta = delta.mean().item() # We might get a delta vector
        
        # Calculate the probability distribution at obs considering opponent action, and consider we did the given action
        action_probabilities = self.actor.probabilities(obs, opponent_action)
        action_log_probabilities = T.log(action_probabilities + 1e-8)
        action_log_probability = action_log_probabilities[..., agent_action]

        # Update the actor network
        self.actor_optimizer.zero_grad()

        actor_entropy = -T.sum(action_log_probabilities * action_probabilities, dim=-1)
        # Perform a correction depending on whether action masking was used
        action_distribution_size = self.actor.distribution_size(obs)
        actor_entropy_corrected = actor_entropy / T.log(T.tensor(action_distribution_size))
        actor_delta = self.cumulative_discount * delta * action_log_probability
        actor_score = actor_delta.mean() + self.actor_entropy_loss_coef * actor_entropy_corrected.mean()
        actor_score.backward()

        if self._actor_gradient_clipping is not None:
            nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), self._actor_gradient_clipping)

        self.actor_optimizer.step()

        if LOGGER.isEnabledFor(logging.DEBUG):
            new_action_probabilities = self.actor.probabilities(obs, opponent_action).detach()
            new_entropy = -T.sum(T.log(new_action_probabilities + 1e-8) * new_action_probabilities, dim=-1)
            LOGGER.debug("Actor was updated using delta %s for action %s and opponent action %s, going from a distribution of %s with entropy %s to a distribution of %s with entropy %s", delta, agent_action, opponent_action, action_probabilities, actor_entropy, new_action_probabilities, new_entropy)

    def value(self, obs: T.Tensor, opponent_action: int | None, critic: QFunction | None = None) -> T.Tensor:
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
            greedy_action = T.argmax(q_so, dim=-1)
            pi = F.one_hot(greedy_action, num_classes=self.action_dim).float()
        elif self._agent_update_style == self.UpdateStyle.UNIFORM:
            n_rows = 1 if opponent_action is not None else self.opponent_action_dim
            pi = T.ones(n_rows, self.action_dim).float() / self.action_dim

        # The policy doesn't need to be transposed since the last dimension is the action dimension.
        # By design, the policy is already transposed (opp_actions X agent_actions).
        # This Einstein summation equation computes the diagonal of a >2D tensor by considering the last two dimensions as matrices.
        return T.einsum("...ii->...i", pi @ q_so.transpose(-2, -1))

    def advantage(self, obs: T.Tensor, next_obs: T.Tensor, reward: float, terminated: bool, agent_action: int | None, opponent_action: int | None, next_obs_opponent_policy: T.Tensor | None, critic: QFunction | None = None) -> T.Tensor:
        """
        Compute the advantage. May consider intrinsic reward if an intrinsic critic was provided.
        The returned tensor is a leaf tensor.
        
        The value is computed as a function of both the Q-function and agent policy:
        - original: `A(s, o, a) = Q(s, o, a) - V(s, o) = Q(s, o, a) - pi(. | s, o).T Q(s, o, .)`
        - alternative: `... = R + omega(. | s') pi(. | s', .).T Q(s', ., .) - pi(. | s, o).T Q(s, o, .)`
        """
        if critic is None:
            critic = self.critic

        if self._alternative_advantage:
            if terminated:
                v_so_next = 0.0
            
            else:
                v_s_next = self.value(next_obs, opponent_action=None, critic=critic).detach().squeeze(0)
                
                if self._opponent_update_style == self.UpdateStyle.EXPECTED_SARSA or self._opponent_update_style == self.UpdateStyle.SARSA:
                    if next_obs_opponent_policy is None:
                        raise ValueError("Sarsa-like updates for the opponent require the opponent's policy on the next observation to be provided")
                    next_opp_pi = next_obs_opponent_policy
                elif self._opponent_update_style == self.UpdateStyle.Q_LEARNING:
                    next_opp_pi = F.one_hot(T.argmin(v_s_next), num_classes=self.opponent_action_dim).unsqueeze(1).float()
                elif self._opponent_update_style == self.UpdateStyle.UNIFORM:
                    next_opp_pi = T.ones(self.opponent_action_dim, 1).float() / self.opponent_action_dim
                else:
                    raise RuntimeError(f"the opponent update style does not have a valid value: {self._opponent_update_style}")
                
                v_so_next = next_opp_pi.T @ v_s_next

            q_soa = reward + self.critic.discount * v_so_next

        else:
            q_soa = critic.q(obs, agent_action, opponent_action).detach()
        
        v_so = self.value(obs, opponent_action, critic=critic).detach()

        advantage = q_soa - v_so
        return advantage
    
    def _update_critics(self, obs: T.Tensor, next_obs: T.Tensor, agent_action: int | None, opponent_action: int | None, next_agent_policy: T.Tensor | str, next_opponent_policy: T.Tensor | str, reward: float, intrinsic_reward: float, terminated: bool) -> tuple[float, float]:
        """Perform an update on the extrinsic and intrinsic critics. The intrinsic critic is only updated if it was specified. Returns the total TD error."""
        # The kwargs that are shared between the critic and the intrinsic critic
        critic_kwargs = {
            "obs": obs,
            "next_obs": next_obs,
            "agent_action": agent_action,
            "opponent_action": opponent_action,
            "next_agent_policy": next_agent_policy,
            "next_opponent_policy": next_opponent_policy,
        }

        extrinsic_td_error = self._critic.update(
            reward=reward,
            terminated=terminated,
            **critic_kwargs,
        )

        # The intrinsic critic not only updates using the intrinsic reward only, but also doesn't consider episode boundaries (is non-episodic), as recommended by the RND paper
        if self._intrinsic_critic is not None:
            intrinsic_td_error = self._intrinsic_critic.update(
                reward=intrinsic_reward,
                terminated=False,
                **critic_kwargs,
            )
        else:
            intrinsic_td_error = T.tensor(0.0)
        
        return extrinsic_td_error.mean().item(), intrinsic_td_error.mean().item()

    def _effective_next_agent_policy(self, agent_action: int | None, agent_will_frameskip: bool, next_obs: T.Tensor) -> T.Tensor | str:
        """Determine what is the agent's effective policy at the next observation `next_obs`."""
        # If the agent will be frameskipped...
        if agent_will_frameskip:
            # ... then we consider they will be doing any of their actions (uniform random policy).
            if agent_action is None:
                next_obs_agent_policy = T.ones(self.action_dim, self.opponent_action_dim).float() / (self.action_dim)
            # ... then we consider the assumed action.
            else:
                next_obs_agent_policy = F.one_hot(T.tensor(agent_action), num_classes=self.action_dim).float().unsqueeze(1).expand(-1, self.opponent_action_dim)
        
        # If the agent could act, then it makes sense for it to have a policy.
        # We have to construct it manually since it is not provided as an argument.
        # It doesn't make sense to provide the policy as argument, since we already have access to it here.
        # On the other hand, we don't have access to the opponent model, hence why it is explicitly provided on that method.
        else:
            # Use the agent's next policy
            if self._agent_update_style == self.UpdateStyle.EXPECTED_SARSA:
                # We don't consider the action that the opponent will perform, this is just the Q-value matrix to provide to the critic method,
                # which will then sort how to consider the opponent's next actions internally.
                next_obs_agent_policy = self.actor.probabilities(next_obs, next_opponent_action=None).detach().squeeze(0).T
            # Enable Q-learning, change agent's policy to greedy
            elif self._agent_update_style == self.UpdateStyle.Q_LEARNING:
                next_obs_agent_policy = "greedy"
            # Change agent's policy to be uniform
            elif self._agent_update_style == self.UpdateStyle.UNIFORM:
                next_obs_agent_policy = "uniform"
            else:
                raise ValueError(f"the agent update style has an invalid value {self._agent_update_style}")
        
        return next_obs_agent_policy

    def _effective_next_opponent_policy(self, opponent_action: int | None, opponent_will_frameskip: bool, next_obs_opponent_policy: T.Tensor | None) -> T.Tensor | str:
        """Determine what is the opponent's effective policy at the next observation `next_obs`. Will return the explicit `next_obs_opponent_policy` back if it should be the one used instead."""
        # If the opponent will be frameskipped...
        if opponent_will_frameskip:
            # ... then we consider they will be doing any of their actions (uniform random policy).
            if opponent_action is None:
                res = T.ones(self.opponent_action_dim, 1).float() / (self.opponent_action_dim)
            # ... then we consider the assumed action.
            else:
                res = F.one_hot(T.tensor(opponent_action), num_classes=self.opponent_action_dim).float().unsqueeze(1)

        else:
            # Keep the opponent's next policy, which should have been provided as an argument
            if self._opponent_update_style == self.UpdateStyle.EXPECTED_SARSA or self._opponent_update_style == self.UpdateStyle.SARSA:
                if next_obs_opponent_policy is None:
                    raise ValueError("Sarsa-like updates for the opponent require the opponent's policy on the next observation to be provided")
                res = next_obs_opponent_policy
            # Enable Q-learning, change opponent's policy to greedy
            elif self._opponent_update_style == self.UpdateStyle.Q_LEARNING:
                res = "greedy"
            # Change opponent's policy to be uniform
            elif self._opponent_update_style == self.UpdateStyle.UNIFORM:
                res = "uniform"
            else:
                raise ValueError(f"the opponent update style has an invalid value {self._opponent_update_style}")
            
        return res

    @dataclass(slots=True)
    class LearnUpdate:
        """Holder of arguments to a `learn()` call, so that we can delay updates to be done in the future, mainly used to implement Sarsa updates"""
        obs: T.Tensor
        next_obs: T.Tensor
        reward: float
        terminated: bool
        truncated: bool
        obs_agent_action: int | None
        obs_opponent_action: int | None
        agent_will_frameskip: bool
        opponent_will_frameskip: bool
        next_obs_opponent_policy: T.Tensor | None
        intrinsic_reward: float

    @dataclass(slots=True)
    class FrameskippedUpdate:
        """Holder of arguments to a learning update, which was frameskippped"""
        obs: T.Tensor
        next_obs: T.Tensor
        reward: float
        terminated: bool
        obs_agent_action: int | None
        obs_opponent_action: int | None
        next_obs_agent_policy: T.Tensor | str
        next_obs_opponent_policy: T.Tensor | str
        intrinsic_reward: float

    def learn(
        self,
        obs: T.Tensor,
        next_obs: T.Tensor,
        reward: float,
        terminated: bool,
        truncated: bool,
        *,
        obs_agent_action: int | None,
        obs_opponent_action: int | None,
        agent_will_frameskip: bool,
        opponent_will_frameskip: bool,
        next_obs_opponent_policy: T.Tensor | None = None,
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
        # If using Sarsa-like updates for the opponent, then we need to learn one time step later
        if self._opponent_update_style == self.UpdateStyle.SARSA and len(self._delayed_updates) < 2:
            self._delayed_updates.append(
                self.LearnUpdate(
                    obs=obs,
                    next_obs=next_obs,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    obs_agent_action=obs_agent_action,
                    obs_opponent_action=obs_opponent_action,
                    agent_will_frameskip=agent_will_frameskip,
                    opponent_will_frameskip=opponent_will_frameskip,
                    next_obs_opponent_policy=None, # We don't care about the next opponent policy; this is because we are going to fill it as a one-hot tensor of the opponent's actual next action
                    intrinsic_reward=intrinsic_reward
                )
            )
            
            # Can't learn yet, we need information of the next time step
            if len(self._delayed_updates) < 2:
                return

            # Learn
            update = self._delayed_updates[0]
            next_obs_opponent_action = self._delayed_updates[1].obs_opponent_action
            next_obs_opponent_policy = F.one_hot(T.tensor(next_obs_opponent_action), num_classes=self.opponent_action_dim).float().unsqueeze(1)

            self.learn(
                obs=update.obs,
                next_obs=update.next_obs,
                reward=update.reward,
                terminated=update.terminated,
                truncated=update.truncated,
                obs_agent_action=update.obs_agent_action,
                obs_opponent_action=update.obs_opponent_action,
                agent_will_frameskip=update.agent_will_frameskip,
                opponent_will_frameskip=update.opponent_will_frameskip,
                next_obs_opponent_policy=next_obs_opponent_policy, # Here, we substitute the future policy with the action the opponent actually took
                intrinsic_reward=update.intrinsic_reward
            )

            # Finish the update and leave
            self._delayed_updates.popleft()
            return

        # Save the TD error in case the caller wants to check it. The TD error is None if no critic update was performed
        self.extrinsic_td_error = None
        self.intrinsic_td_error = None
        self.delta = None

        next_obs_agent_policy_effective = self._effective_next_agent_policy(obs_agent_action, agent_will_frameskip, next_obs)
        next_obs_opponent_policy_effective = self._effective_next_opponent_policy(obs_opponent_action, opponent_will_frameskip, next_obs_opponent_policy)

        # Schedule a critic update
        self.frameskipped_critic_updates.append(
            self.FrameskippedUpdate(
                obs=obs,
                next_obs=next_obs,
                reward=reward,
                terminated=terminated,
                obs_agent_action=obs_agent_action,
                obs_opponent_action=obs_opponent_action,
                next_obs_agent_policy=next_obs_agent_policy_effective,
                next_obs_opponent_policy=next_obs_opponent_policy_effective,
                intrinsic_reward=intrinsic_reward,
            )
        )

        # If the agent will not be frameskipped, then we can perform the updates now, since its last action has been fully resolved.
        # If the game is finished, then we also perform all scheduled updates.
        if (not agent_will_frameskip) or terminated or truncated:
            # Update the reward according to the MaxEnt RL objective.
            # We only do so on the first observation since it's the only ones that is actionable.
            # The last observation is also actionable, but we only receive reward when we reach it.
            # This is important since the last observation may be terminal, in which case it doesn't even make sense to have reward.
            # obs_maxent_reward = self.maxent_reward(self.actor.probabilities(self.frameskipped_critic_updates[0].obs, self.frameskipped_critic_updates[0].obs_opponent_action))
            # The rewards will become tensors, so that the gradient can flow through them (optionally)
            # self.frameskipped_critic_updates[0].reward += obs_maxent_reward.item()

            total_reward: float = sum(update.reward for update in self.frameskipped_critic_updates)
            total_intrinsic_reward: float = sum(update.intrinsic_reward for update in self.frameskipped_critic_updates)

            if self._learn_critic:
                if self._accumulate_at_frameskip:
                    obs_ = self.frameskipped_critic_updates[0].obs
                    next_obs_ = self.frameskipped_critic_updates[-1].next_obs
                    obs_agent_action_ = self.frameskipped_critic_updates[0].obs_agent_action
                    obs_opponent_action_ = self.frameskipped_critic_updates[0].obs_opponent_action # we assume the first opponent action, as it's technically the last one we see
                    next_obs_agent_policy_ = self.frameskipped_critic_updates[-1].next_obs_agent_policy
                    next_obs_opponent_policy_ = self.frameskipped_critic_updates[-1].next_obs_opponent_policy

                    # This will detach the reward from the computational graph, we only care about that for the actor update
                    self.extrinsic_td_error, self.intrinsic_td_error = self._update_critics(obs_, next_obs_, obs_agent_action_, obs_opponent_action_, next_obs_agent_policy_, next_obs_opponent_policy_, total_reward, total_intrinsic_reward, terminated)

                else:
                    # They were None previously. They have None only when an update doesn't occur,
                    # but now that we are going to perform accumulations here we set them to an appropriate start value.
                    self.extrinsic_td_error = 0.0
                    self.intrinsic_td_error = 0.0
                    # Perform the frameskipped updates assuming no discounting (that's pretty much what's special about frameskipped updates)
                    previous_discount = self._critic.discount
                    self._critic.discount = 1.0

                    # We perform the updates in reverse order so that the future reward is propagated back to the oldest retained updates
                    for frameskipped_update in reversed(self.frameskipped_critic_updates):
                        extrinsic_td_error, intrinsic_td_error = self._update_critics(
                            obs=frameskipped_update.obs,
                            next_obs=frameskipped_update.next_obs,
                            agent_action=frameskipped_update.obs_agent_action,
                            opponent_action=frameskipped_update.obs_opponent_action,
                            next_agent_policy=frameskipped_update.next_obs_agent_policy,
                            next_opponent_policy=frameskipped_update.next_obs_opponent_policy,
                            reward=frameskipped_update.reward,
                            intrinsic_reward=frameskipped_update.intrinsic_reward,
                            terminated=frameskipped_update.terminated,
                        )
                        self.extrinsic_td_error += extrinsic_td_error
                        self.intrinsic_td_error += intrinsic_td_error

                    self._critic.discount = previous_discount

            # The agent only performed an action at the beginning of frameskipping. It doesn't make sense to update at any other observation.
            if self._learn_actor:
                obs_ = self.frameskipped_critic_updates[0].obs
                next_obs_ = self.frameskipped_critic_updates[-1].next_obs
                obs_agent_action_ = self.frameskipped_critic_updates[0].obs_agent_action
                obs_opponent_action_ = self.frameskipped_critic_updates[0].obs_opponent_action
                next_obs_opponent_policy_ = self.frameskipped_critic_updates[-1].next_obs_opponent_policy
                if isinstance(next_obs_opponent_policy_, str):
                    next_obs_opponent_policy_ = None # the next obs opponent policy will be built within the advantage() method in cases where it should be None (Q_LEARNING and UNIFORM updates)

                extrinsic_advantage = self.advantage(obs_, next_obs_, total_reward, terminated, obs_agent_action_, obs_opponent_action_, next_obs_opponent_policy_, critic=self.critic)
                if self.intrinsic_critic is not None:
                    intrinsic_advantage = self.advantage(obs_, next_obs_, total_intrinsic_reward, False, obs_agent_action_, obs_opponent_action_, next_obs_opponent_policy_, critic=self.intrinsic_critic)
                else:
                    intrinsic_advantage = 0.0

                if obs_agent_action_ is None:
                    raise RuntimeError("the agent's action at the current observation should be something, not 'None'")
                self._update_actor(obs_, obs_opponent_action_, obs_agent_action_, extrinsic_advantage + intrinsic_advantage)

            self.frameskipped_critic_updates.clear()

        if self.policy_cumulative_discount:
            self.cumulative_discount *= self.critic.discount

        if terminated or truncated:
            self.cumulative_discount = 1.0
            self._last_valid_agent_action = None
            self._last_valid_opponent_action = None
        
    @property
    def actor(self) -> ActorNetwork:
        return self._actor
    
    @property
    def critic(self) -> QFunction:
        return self._critic

    @property
    def intrinsic_critic(self) -> QFunction | None:
        return self._intrinsic_critic

    @property
    def actor_learning_rate(self) -> float:
        return self.actor_optimizer.param_groups[0]["lr"]

    @actor_learning_rate.setter
    def actor_learning_rate(self, value: float):
        self.actor_optimizer.param_groups[0]["lr"] = value
    
    @property
    def critic_learning_rate(self) -> float:
        return self._critic.learning_rate
    
    @critic_learning_rate.setter
    def critic_learning_rate(self, value: float):
        self._critic.learning_rate = value

    @property
    def maxent(self) -> float:
        """The coefficient for the entropy term of the MaxEnt RL objective, which encourages maximization of the policy's expected entropy."""
        return self._maxent

    @maxent.setter
    def maxent(self, value: float):
        self._maxent = value
    
    @property
    def maxent_gradient_flow(self) -> bool:
        """Whether to let the gradient flow through the entropy term in the actor loss."""
        return self._maxent_gradient_flow

    @maxent_gradient_flow.setter
    def maxent_gradient_flow(self, value: bool):
        self._maxent_gradient_flow = value

    @property
    def learn_actor(self) -> bool:
        """Whether the actor is being learnt."""
        return self._learn_actor

    @learn_actor.setter
    def learn_actor(self, value: bool):
        self._learn_actor = value
    
    @property
    def learn_critic(self) -> bool:
        """Whether the critic is being learnt."""
        return self._learn_critic
    
    @learn_critic.setter
    def learn_critic(self, value: bool):
        self._learn_critic = value

    @property
    def agent_update_style(self) -> UpdateStyle:
        return self._agent_update_style

    @agent_update_style.setter
    def agent_update_style(self, value: UpdateStyle):
        self._agent_update_style = value
    
    @property
    def opponent_update_style(self) -> UpdateStyle:
        return self._opponent_update_style
    
    @opponent_update_style.setter
    def opponent_update_style(self, value: UpdateStyle):
        self._opponent_update_style = value
