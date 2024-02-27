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
        **kwargs,
    ):
        a2c_kwargs, actor_kwargs, critic_kwargs = extract_sub_kwargs(kwargs, ("a2c", "actor", "critic"), strict=True)

        self.learner = A2CLambdaLearner(
            actor=ActorNetwork(
                obs_dim=observation_space_size,
                action_dim=action_space_size,
                **actor_kwargs,
            ),
            critic=CriticNetwork(
                obs_dim=observation_space_size,
                **critic_kwargs,
            ),
            **a2c_kwargs,
        ) if learner is None else learner

        self.actor = learner.actor
        self.critic = learner.critic

        self.actor_critic = AggregateModule({
            "actor": self.actor,
            "critic": self.critic,
        })

        self.current_observation = None

    def act(self, obs, info: dict) -> "any":
        self.current_observation = obs
        return self.learner.sample_action(obs)

    def update(self, next_obs, reward: float, terminated: bool, truncated: bool, info: dict):
        self.learner.learn(self.current_observation, next_obs, reward, terminated)

    def extract_policy(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        model = deepcopy(self.actor_critic.actor)

        def internal_policy(obs):
            obs = torch.from_numpy(obs).float().unsqueeze(0)
            probs = model(obs)

            return Categorical(probs=probs).sample().item()

        return super()._extract_policy(env, internal_policy)
    
    @property
    def model(self) -> nn.Module:
        return self.actor_critic