from agents.base import FootsiesAgentBase
from typing import Any, Iterable


class FootsiesAgent(FootsiesAgentBase):
    def __init__(self):
        self.current_observation = None

    def act(self, obs) -> Any:
        self.current_observation = obs


    def update(self, obs, reward: float):
        next_observation = obs
        ...

    def learn_model(self, play_data: "Iterable[tuple[Any, Any, Any, float]]"):
        """Learn the environment model by observing human play data (sequence of state, action, next state and reward tuples)"""
        for obs, action, next_obs, reward in play_data:
            if self.is_primitive(action):
                ...


    def is_primitive(self, action) -> bool:
        raise NotImplementedError

    def load(self, folder_path: str):
        ...

    def save(self, folder_path: str):
        ...