from agents.ql.ql import QTable
from agents.base import FootsiesAgentBase
from gymnasium import Env
from typing import Callable, Tuple
from agents.utils import extract_sub_kwargs


class FootsiesAgent(FootsiesAgentBase):
    def __init__(
        self,
        observation_space_size: int,
        action_space_size: int,
        q_table: QTable = None,
        **kwargs,
    ):
        q_table_kwargs, = extract_sub_kwargs(kwargs, ("q_table",))

        if q_table is None:
            self.q_table = QTable(
                action_dim=action_space_size,
                **q_table_kwargs,
            )
        else:
            self.q_table = q_table

        self.current_obs = None
        self.current_action = None

    def act(self, obs, info: dict) -> "any":
        self.current_obs = obs
        self.current_action = self.q_table.sample_action(obs)
        return self.current_action

    def update(self, next_obs, reward: float, terminated: bool, truncated: bool, info: dict):
        self.q_table.update(self.current_obs, self.current_action, reward, next_obs, terminated)

    def load(self, folder_path: str):
        raise NotImplementedError("loading not supported")

    def save(self, folder_path: str):
        raise NotImplementedError("saving not supported")

    def extract_policy(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        raise NotImplementedError("policy extraction not supported")