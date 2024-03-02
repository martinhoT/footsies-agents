import torch
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

        # Due to the way the gradients are set up, we want the optimizer to maximize (i.e., leave the gradients unchanged)
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

    def sample_action(self, obs: torch.Tensor) -> int:
        """Sample an action from the actor. A training step starts with `act()`, followed immediately by an environment step and `update()`"""    
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

    def _update_critic(self, obs: torch.Tensor, delta: float):
        self.critic_optimizer.zero_grad()

        critic_score = self.critic(obs)
        critic_score.backward()
        with torch.no_grad():
            for critic_trace, parameter in zip(self.critic_traces, self.critic.parameters()):
                critic_trace.copy_(self.discount * self.critic_lambda * critic_trace + parameter.grad)
                parameter.grad.copy_(delta * critic_trace)

        self.critic_optimizer.step()

    def _update_actor(self, delta: float):
        self.actor_optimizer.zero_grad()

        actor_score = self.cumulative_discount * self.action_distribution.log_prob(self.action)
        actor_score.backward(retain_graph=True)
        with torch.no_grad():
            for actor_trace, parameter in zip(self.actor_traces, self.actor.parameters()):
                actor_trace.copy_(self.discount * self.actor_lambda * actor_trace + parameter.grad)
                parameter.grad.copy_(delta * actor_trace * (1 - self.actor_entropy_loss_coef))
        # Add the entropy score gradient
        entropy_score = self.actor_entropy_loss_coef * self.action_distribution.entropy()
        entropy_score.backward()

        self.actor_optimizer.step()

    # TODO: if we use linear function approximation, could we do True Online TD(lambda)?
    def learn(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool):
        """Update the actor and critic networks in this environment step. Should be preceded by an environment interaction with `act()`"""
        # Compute the TD delta
        with torch.no_grad():
            if terminated:
                target = reward
            else:
                target = reward + self.discount * self.critic(next_obs)
            
            delta = (target - self.critic(obs)).item()

        self.delta = delta

        # Update the networks.
        # We perform policy improvement first and then policy evaluation (at_policy_improvement begins True).
        if self.at_policy_improvement:
            self._update_actor(delta)
            self._step_policy_iteration()
        # We don't perform elif since we may want to perform both policy improvement and evaluation in the same step (e.g. if policy_improvement_steps == 1)
        if not self.at_policy_improvement:
            self._update_critic(obs, delta)
            self._step_policy_iteration()

        if self.policy_cumulative_discount:
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
        action_dim: int,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        **kwargs,
    ):
        """Imitation learning applied to a policy"""
        optimizer_kwargs, = extract_sub_kwargs(kwargs, ("optimizer",))
        
        self.policy = policy
        self.action_dim = action_dim
        self.optimizer = optimizer(self.policy.parameters(), maximize=False, **optimizer_kwargs)
        # NOTE: because of this loss, combinatory actions are not supported
        self.loss_fn = nn.KLDivLoss(reduction="batchmean")

    def action_as_onehot(self, action: torch.Tensor) -> torch.Tensor:
        """Transform the action tensor into a one-hot encoded version"""
        return torch.nn.functional.one_hot(action, num_classes=self.action_dim).float()

    def action_as_combination(self, action: torch.Tensor) -> torch.Tensor:
        """Transform the action tensor into a tensor where actions are treated as being combinatory"""
        return torch.hstack([(action & 2**i) != 0 for i in range(self.action_dim)]).float()
                            
    def learn(self, obs: torch.Tensor, action: torch.Tensor, frozen_representation: bool = False) -> float:
        """Update policy by imitating the action at the given observation. Returns the loss"""
        self.optimizer.zero_grad()

        if frozen_representation:
            with torch.no_grad():
                rep = self.policy.representation(obs)
            probs = self.policy.from_representation(rep)

        else:
            probs = self.policy(*obs)

        loss = self.loss_fn(probs.log(), action)
        loss.backward()

        self.optimizer.step()

        return loss.item()