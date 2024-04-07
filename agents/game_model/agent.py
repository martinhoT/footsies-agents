import os
import torch
from torch import nn
from agents.base import FootsiesAgentBase
from agents.action import ActionMap
from gymnasium import Env
from typing import Callable, Tuple, Literal
from agents.game_model.game_model import GameModel


# NOTE: trained by example
# NOTE: player 1 is assumed to be the agent
class GameModelAgent(FootsiesAgentBase):
    def __init__(
        self,
        game_model: GameModel,
    ):
        self._game_model = game_model

        self._last_valid_p1_action = 0
        self._last_valid_p2_action = 0

        self._p1_current_special = None

        self.cumulative_loss = 0
        self.cumulative_loss_n = 0
        self.cumulative_loss_guard = 0
        self.cumulative_loss_move = 0
        self.cumulative_loss_move_progress = 0
        self.cumulative_loss_position = 0

    def act(self, obs: torch.Tensor, info: dict) -> "any":
        return 0

    def update_with_simple_actions(self, obs: torch.Tensor, p1_action: int | None, p2_action: int | None, next_obs: torch.Tensor):
        """Perform an update with the given simple actions, useful to avoid recomputing them."""
        guard_loss, move_loss, move_progress_loss, position_loss = self._game_model.update(obs, p1_action, p2_action, next_obs)

        self.cumulative_loss_guard += guard_loss
        self.cumulative_loss_move += move_loss
        self.cumulative_loss_move_progress += move_progress_loss
        self.cumulative_loss_position += position_loss
        self.cumulative_loss = guard_loss + move_loss + move_progress_loss + position_loss
        self.cumulative_loss_n += 1

    def update(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict):
        # We treat both players equally, to guarantee the Markov property
        p1_action = next_info["p1_simple"]
        p2_action = next_info["p2_simple"]
        self.update_with_simple_actions(obs, p1_action, p2_action, next_obs)

    # This is the only evaluation function that clears the denominator cumulative_loss_n
    def evaluate_average_loss_and_clear(self) -> float:
        res = (
            self.cumulative_loss / self.cumulative_loss_n
        ) if self.cumulative_loss_n != 0 else 0
        
        self.cumulative_loss = 0
        self.cumulative_loss_n = 0

        return res
    
    def evaluate_average_loss_guard(self) -> float:
        res = (
            self.cumulative_loss_guard / self.cumulative_loss_n
        ) if self.cumulative_loss_n != 0 else 0

        self.cumulative_loss_guard = 0

        return res

    def evaluate_average_loss_move(self) -> float:
        res = (
            self.cumulative_loss_move / self.cumulative_loss_n
        ) if self.cumulative_loss_n != 0 else 0

        self.cumulative_loss_move = 0

        return res
    
    def evaluate_average_loss_move_progress(self) -> float:
        res = (
            self.cumulative_loss_move_progress / self.cumulative_loss_n
        ) if self.cumulative_loss_n != 0 else 0

        self.cumulative_loss_move_progress = 0

        return res
    
    def evaluate_average_loss_position(self) -> float:
        res = (
            self.cumulative_loss_position / self.cumulative_loss_n
        ) if self.cumulative_loss_n != 0 else 0

        self.cumulative_loss_position = 0

        return res

    def load(self, folder_path: str):
        model_path = os.path.join(folder_path, "model_weights.pth")
        self.game_model.network.load_state_dict(torch.load(model_path))

    def save(self, folder_path: str):
        model_path = os.path.join(folder_path, "model_weights.pth")
        torch.save(self.game_model.network.state_dict(), model_path)        

    def extract_policy(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        return lambda s: None

    @property
    def game_model(self) -> GameModel:
        return self._game_model
