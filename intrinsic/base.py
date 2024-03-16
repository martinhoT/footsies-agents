import torch
from abc import ABC, abstractmethod


class IntrinsicRewardScheme(ABC):
    """Scheme that adds intrinsic reward to the environment's extrinsic reward signal."""

    @abstractmethod
    def update_and_reward(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict) -> float:
        """Update the scheme's internal state based on the current observation, and return the intrinsic reward associated with this transition."""

    @staticmethod
    @abstractmethod
    def basic() -> "IntrinsicRewardScheme":
        """A basic version of this intrinsic reward scheme, with good defaults for its arguments."""