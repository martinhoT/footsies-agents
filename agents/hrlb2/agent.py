from agents.base import FootsiesAgentBase
from typing import Any


class FootsiesAgent(FootsiesAgentBase):
    def act(self, obs) -> Any:
        pass

    def update(self, obs, reward: float):
        pass
