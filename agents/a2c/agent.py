import torch
from torch import nn
from torch.distributions import Categorical
from copy import deepcopy
from agents.base import FootsiesAgentTorch
from gymnasium import Env
from typing import Callable, Tuple
from agents.a2c.a2c import A2CLambdaLearner, ActorNetwork, CriticNetwork
from agents.utils import extract_sub_kwargs
from agents.torch_utils import AggregateModule


class FootsiesAgent(FootsiesAgentTorch):
    def __init__(
        self,
        observation_space_size: int,
        action_space_size: int,
        learner: A2CLambdaLearner = None,
        actor_hidden_layer_sizes_specification: str = "",
        critic_hidden_layer_sizes_specification: str = "",
        actor_hidden_layer_activation_specification: str = "ReLU",
        critic_hidden_layer_activation_specification: str = "ReLU",
        **kwargs,
    ):
        a2c_kwargs, actor_kwargs, critic_kwargs = extract_sub_kwargs(kwargs, ("a2c", "actor", "critic"), strict=True)

        learner = A2CLambdaLearner(
            actor=ActorNetwork(
                obs_dim=observation_space_size,
                action_dim=action_space_size,
                hidden_layer_sizes=[int(n) for n in actor_hidden_layer_sizes_specification.split(",")] if actor_hidden_layer_sizes_specification else [],
                hidden_layer_activation=getattr(nn, actor_hidden_layer_activation_specification),
                **actor_kwargs,
            ),
            critic=CriticNetwork(
                obs_dim=observation_space_size,
                hidden_layer_sizes=[int(n) for n in critic_hidden_layer_sizes_specification.split(",")] if critic_hidden_layer_sizes_specification else [],
                hidden_layer_activation=getattr(nn, critic_hidden_layer_activation_specification),
                **critic_kwargs,
            ),
            **a2c_kwargs,
        ) if learner is None else learner

        self.learner = learner
        self.actor = learner.actor
        self.critic = learner.critic

        self.actor_critic = AggregateModule({
            "actor": self.actor,
            "critic": self.critic,
        })

        self.current_observation = None

        # For logging
        self.cumulative_delta = 0
        self.cumulative_delta_n = 0

    def act(self, obs: torch.Tensor, info: dict) -> "any":
        self.current_observation = obs
        return self.learner.sample_action(obs)

    def update(self, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict):
        self.learner.learn(self.current_observation, next_obs, reward, terminated)
        
        self.cumulative_delta += self.learner.delta
        self.cumulative_delta_n += 1

    def extract_policy(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        model = deepcopy(self.actor_critic.actor)

        def internal_policy(obs):
            probs = model(obs)
            return Categorical(probs=probs).sample().item()

        return super()._extract_policy(env, internal_policy)
    
    @property
    def model(self) -> nn.Module:
        return self.actor_critic
    
    def evaluate_average_delta(self) -> float:
        res = (
            self.cumulative_delta / self.cumulative_delta_n
        ) if self.cumulative_delta_n != 0 else 0

        self.cumulative_delta = 0
        self.cumulative_delta_n = 0

        return res
