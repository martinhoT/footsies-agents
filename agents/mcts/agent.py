from agents.base import FootsiesAgentBase
from typing import Any
from time import monotonic


class FootsiesAgent(FootsiesAgentBase):
    def __init__(self, fps: int=60):
        self.decision_time_window = 1 / fps
        # If we are missing frames, then increase this value which will shorten the decision time window
        self.decision_time_window_penalty = 0

    def act(self, obs) -> Any:
        # Choosing an action (the search step) makes use of the entire available decision time (most of it at least)
        current_time = monotonic()
        # While we have time to search
        while (monotonic() - current_time) < (self.decision_time_window - self.decision_time_window_penalty):
            ...

    def update(self, obs, reward: float):
        pass