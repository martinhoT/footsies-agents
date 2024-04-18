import os
import torch
import logging
from torch import nn
from agents.base import FootsiesAgentBase
from agents.action import ActionMap
from gymnasium import Env
from typing import Callable, Tuple, Literal
from agents.game_model.game_model import GameModel
from copy import deepcopy
from collections import deque
from dataclasses import dataclass

LOGGER = logging.getLogger("main.game_model")


@dataclass(slots=True)
class ScheduledUpdate:
    life:       int
    obs:        torch.Tensor
    p1_action:  int | None
    p2_action:  int | None


# NOTE: trained by example
# NOTE: player 1 is assumed to be the agent
class GameModelAgent(FootsiesAgentBase):
    def __init__(
        self,
        game_model: GameModel,
        steps_n: list[int] | None = None,
    ):
        if steps_n is None:
            steps_n = [1]

        self._game_models: list[tuple[int, GameModel]] = []
        for step_n in sorted(steps_n):
            self._game_models.append((step_n, deepcopy(game_model)))
        # Each game model can predict a different number of timesteps into the future, so we need to treat updates differently.
        # Therefore, we have a list of scheduled updates with an element for each game model, and that element is a deque
        # with incomplete updates (those that require `next_obs`).
        self._scheduled_updates: list[deque[ScheduledUpdate]] = [deque([]) for _ in self._game_models]

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
        guard_loss, move_loss, move_progress_loss, position_loss = 0.0, 0.0, 0.0, 0.0
        there_was_update = False

        for (step_n, g), g_updates in zip(self._game_models, self._scheduled_updates):
            scheduled_update = ScheduledUpdate(
                life=step_n,
                obs=obs,
                p1_action=p1_action,
                p2_action=p2_action,
            )
            g_updates.append(scheduled_update)

            for scheduled_update in g_updates:
                scheduled_update.life -= 1

            u = g_updates[0]

            if u.life < 0:
                LOGGER.warning("A scheduled update's life was less than 0, it must have been ignored or some mismanagement happened! Update: %s", u)

            if u.life == 0:
                gl, ml, mpl, pl = g.update(u.obs, u.p1_action, u.p2_action, next_obs)
                guard_loss += gl
                move_loss += ml
                move_progress_loss += mpl
                position_loss += pl
                there_was_update = True

                g_updates.popleft()

        if there_was_update:
            self.cumulative_loss_guard += guard_loss
            self.cumulative_loss_move += move_loss
            self.cumulative_loss_move_progress += move_progress_loss
            self.cumulative_loss_position += position_loss
            self.cumulative_loss += guard_loss + move_loss + move_progress_loss + position_loss
            self.cumulative_loss_n += 1

    def update(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict):
        # We treat both players equally, to guarantee the Markov property
        p1_action = next_info["p1_simple"]
        p2_action = next_info["p2_simple"]
        self.update_with_simple_actions(obs, p1_action, p2_action, next_obs)

    @torch.no_grad
    def predict(self, obs: torch.Tensor, p1_action: int, p2_action: int, n: int) -> tuple[torch.Tensor, int]:
        """
        Predict the next observation from the current one, which is `n` timesteps ahead.

        Returns
        -------
        - `next_obs`: the predicted next observation
        - `steps`: how many timesteps in the future `next_obs` is from `obs`
        """
        # Get the game model that fulfills the `n` request the greatest without overflowing
        step_n, game_model = max(((step_n, g) for step_n, g in self._game_models if step_n <= n), key=lambda t: t[0])

        return game_model.predict(obs, p1_action, p2_action), step_n

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
        for i, (_, g) in enumerate(self._game_models):
            model_path = os.path.join(folder_path, f"game_model_{i}.pth")
            g.network.load_state_dict(torch.load(model_path))

    def save(self, folder_path: str):
        for i, (_, g) in enumerate(self._game_models):
            model_path = os.path.join(folder_path, f"game_model_{i}.pth")
            torch.save(g.network.state_dict(), model_path)        

    def extract_opponent(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        return lambda s: None

    @property
    def game_models(self) -> list[tuple[int, GameModel]]:
        return self._game_models

    @property
    def min_resolution(self) -> int:
        """The minimum prediction resolution (i.e. there only exist models that predict from `n` timesteps ahead upwards)."""
        return min(steps_n for steps_n, _ in self._game_models)
