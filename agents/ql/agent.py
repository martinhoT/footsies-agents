import os
from agents.ql.ql import QFunctionTable
from agents.base import FootsiesAgentBase
from gymnasium import Env
from typing import Any, Callable, Tuple
from agents.utils import extract_sub_kwargs


class QLAgent(FootsiesAgentBase):
    def __init__(
        self,
        observation_space_size: int,
        action_space_size: int,
        q_table: QFunctionTable | None = None,
        **kwargs,
    ):
        q_table_kwargs, = extract_sub_kwargs(kwargs, ("q_table",))

        if q_table is None:
            self.q_table = QFunctionTable(
                action_dim=action_space_size,
                **q_table_kwargs,
            )
        else:
            self.q_table = q_table

        self.current_action = None

    def act(self, obs, info: dict) -> Any:
        self.current_action = self.q_table.sample_action_best(obs)
        return self.current_action

    def update(self, obs, next_obs, reward: float, terminated: bool, truncated: bool, info, next_info: dict):
        self.q_table.update(obs, reward, next_obs, terminated, self.current_action)

    def load(self, folder_path: str):
        path = os.path.join(folder_path, "q")
        self.q_table.load(path)

    def save(self, folder_path: str):
        path = os.path.join(folder_path, "q")
        self.q_table.save(path)

    def extract_opponent(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        raise NotImplementedError("policy extraction not supported")