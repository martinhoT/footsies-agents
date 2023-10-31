from typing import Any

from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatten_space
from agents.base import FootsiesAgentBase
from torch import nn
import torch
import numpy as np


class Network(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(n_observations, 800),
            nn.LeakyReLU(),
            nn.Linear(800, 600),
            nn.LeakyReLU(),
            nn.Linear(600, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.Linear(100, n_actions)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        output = self.layers(x)
        return output


class FootsiesAgent(FootsiesAgentBase):
    def __init__(self,
        observation_space: Space,
        action_space: Space
    ):
        self.observations_length = flatten_space(observation_space).shape[0]
        self.actions_length = flatten_space(action_space).shape[0]
        self.network = Network(self.observations_length, self.actions_length)

    def act(self, obs) -> Any:
        if not isinstance(obs, torch.Tensor):
            obs = (
                torch.tensor(obs, dtype=torch.float32).to(self.device).reshape((1, -1))
            )
        
        output = self.network(obs)

        # Transliteration from the Java code, I don't know what the magic numbers are for
        sum_temp = np.sum(np.exp(output - 10))
        action_max = np.argmax(output)

        



    def update(self, next_obs, reward: float, terminated: bool, truncated: bool):
        pass
