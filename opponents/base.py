from abc import ABC, abstractmethod


class Opponent(ABC):

    @abstractmethod
    def act(self, obs: dict, info: dict) -> tuple[bool, bool, bool]:
        """Perform an action given an observation and info."""
    
    @abstractmethod
    def reset(self):
        """Reset internal state, done at the beginning of every episode."""


class OpponentManager(ABC):

    @abstractmethod
    def update_at_episode(self, game_result: float) -> bool:
        """Update the manager after every episode. Returns whether an opponent change was done."""

    @property
    @abstractmethod
    def current_opponent(self) -> Opponent | None:
        """The opponent being currently used for training. If `None`, then the in-game bot is being used rather than a custom one."""

    @property
    @abstractmethod
    def exhausted(self) -> bool:
        """Whether the opponent pool has been exhausted and there is no opponent to play against."""
