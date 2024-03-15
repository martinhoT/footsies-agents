from abc import ABC, abstractmethod
from typing import Callable


class OpponentManager:

    @abstractmethod
    def update_at_episode(self, game_result: float) -> bool:
        """Update the manager after every episode. Returns whether an opponent change was done."""

    @property
    @abstractmethod
    def current_opponent(self) -> Callable[[dict], tuple[bool, bool, bool]]:
        """The opponent being currently used for training. If `None`, then the in-game bot is being used rather than a custom one."""

    @property
    @abstractmethod
    def exhausted(self) -> bool:
        """Whether the opponent pool has been exhausted and there is no opponent to play against."""
