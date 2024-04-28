from typing import Protocol, Callable, Any
from agents.base import FootsiesAgentBase

class ModelInit(Protocol):
    def __call__(self, observation_space_size: int, action_space_size: int, **kwargs) -> tuple[FootsiesAgentBase, dict[str, list]]:
        ...