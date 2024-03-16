import logging
from abc import ABC, abstractmethod
from collections import deque
from footsies_gym.moves import FootsiesMove
from itertools import cycle
from agents.action import ActionMap
from opponents.base import OpponentManager
from typing import Callable
from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger("main.curriculum")


class CurriculumManager(OpponentManager):
    def __init__(
        self,
        win_rate_threshold: float = 0.7,
        win_rate_over_episodes: int = 100,
        log_dir: str = None,
    ):
        self._win_rate_threshold = win_rate_threshold
    
        self._current_opponent_idx = 0
        self._opponents = sorted([
            Idle(),
            Backer(),
            Spammer(),
            ForwardSpammer(),
            NSpecialSpammer(),
            BSpecialSpammer(),
            WhiffPunisher(),
        ], key=lambda o: o.difficulty)
        
        # An array of booleans that tracks the most recent wins of the agent (True if win, False otherwise)
        self._agent_wins = deque([], maxlen=win_rate_over_episodes)
        # The total number of episodes since the last switch
        self._current_episodes = 0

        # This is the tracker of episodes so far, for logging
        self._episode = 0

        self._summary_writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None
    
    def _is_next_opponent_ready(self):
        """Check whether the agent has surpassed the current opponent enough."""
        return len(self._agent_wins) == self._agent_wins.maxlen and self.current_recent_win_rate > self._win_rate_threshold
            
    def _advance(self) -> "CurriculumOpponent":
        """Advance to the next opponent. The next opponent to which this method advances is returned. If the returned opponent is `None`, then the curriculum is complete."""
        self._agent_wins.clear()
        self._current_episodes = 0

        if self._current_opponent_idx + 1 >= len(self._opponents):
            return None

        self._current_opponent_idx += 1
        new_opponent = self._opponents[self._current_opponent_idx]
        return new_opponent
    
    def update_at_episode(self, game_result: float):
        if self.exhausted:
            raise ValueError("can't perform another update while the opponent manager is exhausted")

        self._agent_wins.append(game_result == 1.0)
        self._episode += 1
        self._current_episodes += 1

        opponent_change = False

        if self._is_next_opponent_ready():
            previous_opponent = self._opponents[self._current_opponent_idx]
            previous_wins = sum(self._agent_wins)
            previous_episodes = self._current_episodes

            new_opponent = self._advance()

            LOGGER.info(f"Agent has surpassed opponent '{previous_opponent.__class__.__name__}' with a win rate of {previous_wins / self._agent_wins.maxlen:%} over the recent {self._agent_wins.maxlen} after {previous_episodes} episodes. Switched to {new_opponent.__class__.__name__}")
            opponent_change = True

        # Logging
        if self._summary_writer is not None:
            self._summary_writer.add_scalar(
                "Performance/Win rate against current curriculum opponent",
                self.current_recent_win_rate,
                self._episode,
            )

        return opponent_change

    @property
    def current_opponent(self) -> Callable[[dict], tuple[bool, bool, bool]]:
        return self._opponents[self._current_opponent_idx].act

    @property
    def exhausted(self) -> bool:
        return self._current_opponent_idx >= len(self._opponents)
    
    @property
    def current_recent_win_rate(self) -> float:
        return sum(self._agent_wins) / len(self._agent_wins) if self._agent_wins else 0.5


class CurriculumOpponent(ABC):

    @abstractmethod
    def act(self, obs: dict) -> tuple[bool, bool, bool]:
        """The method by which the opponent performs actions"""

    @property
    @abstractmethod
    def difficulty(self) -> int:
        pass


class Idle(CurriculumOpponent):

    def act(self, obs: dict) -> tuple[bool, bool, bool]:
        return (False, False, False)
    
    @property
    def difficulty(self) -> int:
        return 0
    

class Backer(CurriculumOpponent):

    def act(self, obs: dict) -> tuple[bool, bool, bool]:
        return (False, True, False)

    @property
    def difficulty(self) -> int:
        return 0


class Spammer(CurriculumOpponent):
    def __init__(self):
        self._action_cycle = cycle([
            (False, False, True),
            (False, False, False),
        ])

    def act(self, obs: dict) -> tuple[bool, bool, bool]:
        return next(self._action_cycle)
    
    @property
    def difficulty(self) -> int:
        return 1


class ForwardSpammer(CurriculumOpponent):
    def __init__(self):
        self._action_cycle = cycle([
            (True, False, True),
            (True, False, False),
            (True, False, False),
            (True, False, False),
        ])
    
    def act(self, obs: dict) -> tuple[bool, bool, bool]:
        return next(self._action_cycle)

    @property
    def difficulty(self) -> int:
        return 1


class NSpecialSpammer(CurriculumOpponent):
    def __init__(self):
        self._action_cycle = cycle([
            *([(False, False, True)] * 60),
            (False, False, False),
        ])
    
    def act(self, obs: dict) -> tuple[bool, bool, bool]:
        return next(self._action_cycle)
    
    @property
    def difficulty(self) -> int:
        return 2


class BSpecialSpammer(CurriculumOpponent):
    def __init__(self):
        self._action_cycle = cycle([
            *([(False, False, True)] * 60),
            (True, False, False),
        ])
    
    def act(self, obs: dict) -> tuple[bool, bool, bool]:
        return next(self._action_cycle)
    
    @property
    def difficulty(self) -> int:
        return 2


class WhiffPunisher(CurriculumOpponent):
    def __init__(self, keepout_distance: float = 2.624):
        self._keepout_distance = keepout_distance
        self._action_queue = deque([])

    def act(self, obs: dict) -> tuple[bool, bool, bool]:
        # If there were actions that was previously queued, perform them
        if self._action_queue:
            return self._action_queue.popleft()

        agent_move: FootsiesMove = ActionMap.move_from_move_index(obs["move"][0])
        agent_move_frame: int = obs["move_frame"][0]
    
        # If we hit the agent, finish them off
        if agent_move == FootsiesMove.DAMAGE:
            return (False, False, True)

        # Move towards the agent by default if distance between players is too great
        distance_between_players = obs["position"][1] - obs["position"][0]
        if distance_between_players > self._keepout_distance:
            return (True, False, False)

        if agent_move in (FootsiesMove.N_ATTACK, FootsiesMove.B_ATTACK, FootsiesMove.N_SPECIAL, FootsiesMove.B_SPECIAL):
            # If the agent is in recovery and we can hit them before they recover...
            if agent_move.in_recovery(agent_move_frame) and agent_move_frame + FootsiesMove.N_ATTACK.value.startup < agent_move.value.duration:
                # ... punish them (we also need to have an idle input so that we are not charging a special move, and can perform the full punish)
                self._action_queue.append((False, False, False))
                return (False, False, True)

            # If the agent is beginning to perform the move...
            else:
                # ... just block
                return (False, True, False)
        
        # Don't get too close to the opponent
        return (False, True, False)

    @property
    def difficulty(self):
        return 2