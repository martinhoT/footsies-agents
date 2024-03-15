from abc import ABC, abstractmethod
from collections import deque
from footsies_gym.moves import FootsiesMove
from itertools import cycle
from agents.action import ActionMap
from opponents.base import OpponentManager
from typing import Callable


class CurriculumManager(OpponentManager):
    def __init__(self, win_rate_threshold: float = 0.7, min_episodes: int = 100):
        self._win_rate_threshold = win_rate_threshold
        self._min_episodes = min_episodes
    
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
        
        self._agent_wins = 0
        self._episodes = 0
    
    def _is_next_opponent_ready(self):
        """Check whether the agent has surpassed the current opponent enough."""
        return self._episodes >= self._min_episodes and (self._agent_wins / self._episodes) > self._win_rate_threshold
            
    def _advance(self) -> "CurriculumOpponent":
        """Advance to the next opponent. The next opponent to which this method advances is returned. If the returned opponent is `None`, then the curriculum is complete."""
        self._agent_wins = 0
        self._episodes = 0

        if self._current_opponent_idx >= len(self._opponents):
            return None

        new_opponent = self._opponents[self._current_opponent_idx]
        self._current_opponent_idx += 1
        return new_opponent
    
    def update_at_episode(self, game_result: float):
        if self.exhausted:
            raise ValueError("can't perform another update while the opponent manager is exhausted")

        self._agent_wins += (game_result == 1.0)
        self._episodes += 1

        if self._is_next_opponent_ready():
            self._advance()
            return True

        return False

    @property
    def current_oppponent(self) -> Callable[[dict], tuple[bool, bool, bool]]:
        return self._opponents[self._current_opponent_idx].act

    @property
    def exhausted(self) -> bool:
        return self._current_opponent_idx >= len(self._opponents)


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
            # If the agent is in recovery...
            if agent_move.in_recovery(agent_move_frame):
                # ... punish them (we also need to have an idle input so that we are not charging a special move, and can perform the full punish)
                self._action_queue.append((False, False, False))
                return (False, False, True)

            # If the agent is beginning to perform the move...
            else:
                # ... just wait
                return (False, False, False)
        
        # Don't get too close to the opponent
        return (False, True, False)

    @property
    def difficulty(self):
        return 2