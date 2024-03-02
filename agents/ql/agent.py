import numpy as np
from agents.base import FootsiesAgentBase
from gymnasium import Env
from typing import Callable, Tuple


class QTable():
    def __init__(self, lr: float, discount: float):
        self.lr = lr
        self.discount = discount

        self.move_frame_bins = np.linspace(-0.1, 1.1, 5)
        self.position_bins = np.linspace(-1.1, 1.1, 5)
        
        self.table = np.zeros((self.obs_size, self.action_size), dtype=np.float32)
        self.update_frequency_table = np.zeros((self.obs_size, self.action_size), dtype=np.int32)

    def obs_idx(self, obs: np.ndarray) -> int:
        gu1, gu2 = tuple(np.round(obs[0:2] * 3))
        mo1 = np.argmax(obs[2:17])
        mo2 = np.argmax(obs[17:32])
        mf1, mf2 = tuple(np.digitize(obs[32:34], self.move_frame_bins))
        po1, po2 = tuple(np.digitize(obs[34:36], self.position_bins))
        # yikes
        return int(gu1 + 4 * gu2 + 4 * 4 * mo1 + 4 * 4 * 15 * mo2 + 4 * 4 * 15 * 15 * mf1 + 4 * 4 * 15 * 15 * 5 * mf2 + 4 * 4 * 15 * 15 * 5 * 5 * po1 + 4 * 4 * 15 * 15 * 5 * 5 * 5 * po2)

    def update(self, obs: dict, action: int, reward: float, next_obs: dict, terminated: bool) -> float:
        obs = self.obs_idx(obs)
        next_obs = self.obs_idx(next_obs)

        nxt = (self.discount * np.max(self.table[next_obs, :])) if not terminated else 0.0
        td_error = (reward + nxt - self.table[obs, action])
        self.table[obs, action] += self.lr * td_error
        self.update_frequency_table[obs, action] += 1

        return td_error
    
    def sample_action(self, obs: dict) -> int:
        return np.argmax(self.table[self.obs_idx(obs), :])

    @property
    def obs_size(self) -> int:
        return 4**2 * 15**2 * 5**2 * 5**2
    
    @property
    def action_size(self) -> int:
        return 8


class FootsiesAgent(FootsiesAgentBase):
    def __init__(
        self,
        observation_space_size: int,
        action_space_size: int,
    ):
        self.q_table = QTable()

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