from agents.base import FootsiesAgentBase


class FootsiesAgent(FootsiesAgentBase):
    def act(self, obs) -> "any":
        ...

    def update(self, next_obs, reward: float, terminated: bool, truncated: bool):
        ...

    def load(self, folder_path: str):
        ...

    def save(self, folder_path: str):
        ...
