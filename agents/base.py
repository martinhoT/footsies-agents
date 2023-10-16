from abc import ABC, abstractmethod
from typing import Any


class FootsiesAgentBase(ABC):
    @abstractmethod
    def act(self, obs) -> Any:
        """Get the chosen action for the currently observed environment state"""
        pass

    @abstractmethod
    def update(self, next_obs, reward: float):
        """After acting in the environment, process the perceived observation and reward"""
        pass
