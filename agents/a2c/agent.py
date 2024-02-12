from agents.base import FootsiesAgentBase
from gymnasium import Env
from typing import Callable, Tuple


class FootsiesAgent(FootsiesAgentBase):
    def act(self, obs) -> "any":
        ...

    def update(self, next_obs, reward: float, terminated: bool, truncated: bool, info: dict):
        ...

    # def preprocess(self, env: Env):
    #     ...

    def load(self, folder_path: str):
        ...

    def save(self, folder_path: str):
        ...

    def extract_policy(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        ...