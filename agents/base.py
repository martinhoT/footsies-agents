from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple
from gymnasium import Env
from torch import nn

from agents.utils import wrap_policy


class FootsiesAgentBase(ABC):
    @abstractmethod
    def act(self, obs, info: dict) -> Any:
        """Get the chosen action for the currently observed environment state"""

    @abstractmethod
    def update(self, next_obs, reward: float, terminated: bool, truncated: bool, info: dict):
        """After acting in the environment, process the perceived observation and reward"""

    def preprocess(self, env: Env):
        """Do some preprocessing on the environment before training on it"""

    @abstractmethod
    def load(self, folder_path: str):
        """Load the agent from disk"""

    @abstractmethod
    def save(self, folder_path: str):
        """Save the agent to disk (overwriting an already saved agent at that path)"""

    @abstractmethod
    def extract_policy(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        """
        Extract a policy which can be provided to the FOOTSIES environment as an opponent.

        Subclasses of `FootsiesAgentBase` should implement the `extract_policy` method themselves.
        This method should return `super()._extract_policy(env, wrapped)`, where `wrapped` is the subclass's policy
        as used for training.
        This is relevant since the environment can be wrapped with observation and action wrappers, of which the
        subclass agent is not aware (or at least doesn't need to be).
        The superclass method deals with some of these wrappers to avoid code duplication
        """

    def _extract_policy(
        self, env: Env, internal_policy: Callable
    ) -> Callable[[dict], Tuple[bool, bool, bool]]:
        return wrap_policy(env, internal_policy)


class FootsiesAgentTorch(FootsiesAgentBase):
    @property
    @abstractmethod
    def model(self) -> nn.Module:
        """The PyTorch model used by the agent"""
