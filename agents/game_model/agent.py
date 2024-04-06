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
class FootsiesAgent(FootsiesAgentBase):
    def __init__(
        self,
        game_model: GameModel,
        assume_action_on_frameskip: Literal["last", "any", "none"] = "last",
        p1_simple_action_correction: bool = True,
    ):
        if assume_action_on_frameskip not in ("last", "any", "none"):
            raise ValueError("'assume_action_on_frameskip' must be either 'last', 'any' or 'none'")

        self._game_model = game_model
        self._assume_action_on_frameskip = assume_action_on_frameskip
        self._p1_simple_action_correction = p1_simple_action_correction

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
        # Skip updates in hitstop (i.e. don't predict the time "freeze")
        p1_in_hitstop = ActionMap.is_in_hitstop_torch(obs, True)
        p2_in_hitstop = ActionMap.is_in_hitstop_torch(obs, False)
        if (p1_in_hitstop or p2_in_hitstop) and obs.isclose(next_obs).all():
            return
        
        if p1_action is None:
            if self._assume_action_on_frameskip == "last":
                p1_action = self._last_valid_p1_action
            elif self._assume_action_on_frameskip == "none":
                p1_action = 0
            else:
                p1_action = None
        else:
            self._last_valid_p1_action = p1_action

        if p2_action is None:
            p2_action = self._last_valid_p2_action
            p2_action = 0
        else:
            self._last_valid_p2_action = p2_action

        # Perform a move progress correction, by considering the agent to be performing a special move even as it's attempting its motion (primitive input sequence).
        # We therefore consider the move to be finished, and visible, at the last primitive action.
        if self._p1_simple_action_correction and ActionMap.is_simple_action_special_move(p1_action):
            p1_simple = ActionMap.simple_as_move(p1_action)
            p1_primitive_sequence_length = len(ActionMap.simple_to_primitive(p1_action))
            obs = obs.clone()
            obs[:, 32] = obs[:, 32] * p1_simple.value.duration / (p1_simple.value.duration + p1_primitive_sequence_length - 1)

        guard_loss, move_loss, move_progress_loss, position_loss = self._game_model.update(obs, p1_action, p2_action, next_obs)

        self.cumulative_loss_guard += guard_loss
        self.cumulative_loss_move += move_loss
        self.cumulative_loss_move_progress += move_progress_loss
        self.cumulative_loss_position += position_loss
        self.cumulative_loss = guard_loss + move_loss + move_progress_loss + position_loss
        self.cumulative_loss_n += 1

    def update(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict):
        p1_action, p2_action = ActionMap.simples_from_transition_ori(info, next_info)
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
