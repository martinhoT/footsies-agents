from abc import ABC, abstractmethod
from typing import Any


class FootsiesAgentBase(ABC):
    @abstractmethod
    def act(self, obs) -> Any:
        """Get the chosen action for the currently observed environment state"""

    @abstractmethod
    def update(self, next_obs, reward: float, terminated: bool, truncated: bool):
        """After acting in the environment, process the perceived observation and reward"""

    @abstractmethod
    def load(self, folder_path: str):
        """Load the agent from disk"""

    @abstractmethod
    def save(self, folder_path: str):
        """Save the agent to disk (overwriting an already saved agent at that path)"""
