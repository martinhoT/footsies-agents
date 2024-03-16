import torch
from intrinsic.base import IntrinsicRewardScheme


class NoveltyTable(IntrinsicRewardScheme):
    def __init__(
        self,
        reward_scale: float = 1.0
    ):
        self.table = {}
        self.reward_scale = reward_scale
    
    def register(self, obs: torch.Tensor):
        o = obs.numpy().tobytes()
        self.table[o] = self.table.get(o, 0) + 1
    
    def query(self, obs: torch.Tensor) -> int:
        o = obs.numpy().tobytes()
        return self.table.get(o, 0)

    def intrinsic_reward(self, obs: torch.Tensor) -> float:
        return self.reward_scale / self.query(obs)


class CountBasedScheme(IntrinsicRewardScheme):
    def __init__(self, table: NoveltyTable):
        self.table = table

    def update_and_reward(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict) -> float:
        self.table.register(obs)
        return self.table.intrinsic_reward(obs)

    @staticmethod
    def basic() -> "IntrinsicRewardScheme":
        table = NoveltyTable(reward_scale=1.0)
        return CountBasedScheme(table)