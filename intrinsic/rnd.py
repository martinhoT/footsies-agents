import torch
from torch import nn
from agents.torch_utils import create_layered_network
from intrinsic.base import IntrinsicRewardScheme
from collections import namedtuple
from typing import Callable
from math import sqrt


class RNDNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        embedding_dim: int,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: type[nn.Module] = nn.Identity,
    ):
        super().__init__()

        self.layers = create_layered_network(obs_dim, embedding_dim, hidden_layer_sizes, hidden_layer_activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.layers(obs)


class RandomNetworkDistillation:
    Transition = namedtuple("Transition", ["obs", "next_obs", "action", "reward_extrinsic", "reward_intrinsic"])
    
    def __init__(
        self,
        rnd_generator: Callable[[], RNDNetwork],
        learning_rate: float = 1e-3,
        reward_normalization: bool = True,
        observation_normalization: bool = True,
        mini_batch_size: int = 1000,
    ):
        if reward_normalization or observation_normalization:
            raise NotImplementedError("reward_normalization and observation_normalization are not supported yet")
    
        self._reward_normalization = reward_normalization
        self._observation_normalization = observation_normalization
        self._mini_batch_size = mini_batch_size

        self._target = rnd_generator()
        self._predictor = rnd_generator()

        self._input_size = self._target.layers[0].in_features
        self._output_size = self._target.layers[-1].out_features

        # Initialize the networks randomly
        # for param in self._target.parameters():
        #     nn.init.uniform_(param.data, -1 / sqrt(self._input_size), 1 / sqrt(self._output_size))
        # for param in self._predictor.parameters():
        #     nn.init.uniform_(param.data, -1 / sqrt(self._input_size), 1 / sqrt(self._output_size))

        self._target.requires_grad_(False)
        self._optimizer = torch.optim.SGD(self._predictor.parameters(), lr=learning_rate)
        self._batch: list[RandomNetworkDistillation.Transition] = []

    def reward(self, obs: torch.Tensor) -> torch.Tensor:
        """The reward as a function of the prediction error for the given observation, between the predictor and the target network."""
        target = self._target(obs)
        prediction = self._predictor(obs)

        return (target - prediction).pow(2).mean()

    def update(self, obs: torch.Tensor, reward_intrinsic: torch.Tensor):
        self._batch.append(reward_intrinsic)
        
        if len(self._batch) >= self._mini_batch_size:
            intrinsic_rewards = torch.vstack(self._batch)

            self._optimizer.zero_grad()

            loss = torch.mean(intrinsic_rewards)
            loss.backward()

            self._optimizer.step()

            self._batch.clear()


class RNDScheme(IntrinsicRewardScheme):
    def __init__(self, rnd: RandomNetworkDistillation):
        self.rnd = rnd
    
    def update_and_reward(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict) -> float:
        # Intrinsic reward is calculated before the update
        intrinsic_reward = self.rnd.reward(next_obs)
        self.rnd.update(next_obs, intrinsic_reward)
        # Only do item() here! We need to let intrinsic_reward be a part of the computational graph in update()
        return intrinsic_reward.item()
    
    @staticmethod
    def basic(obs_dim: int = 36, embedding_dim: int = 8) -> "IntrinsicRewardScheme":
        rnd = RandomNetworkDistillation(
            rnd_generator=lambda: RNDNetwork(
                obs_dim=obs_dim,
                embedding_dim=embedding_dim,
                hidden_layer_sizes=[64, 64],
                hidden_layer_activation=nn.LeakyReLU,
            ),
            learning_rate=1e-3,
            reward_normalization=False,
            observation_normalization=False,
            mini_batch_size=1,
        )
        return RNDScheme(rnd)