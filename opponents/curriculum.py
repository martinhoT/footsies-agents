import torch
import logging
import csv
from torch import nn
from abc import abstractmethod
from collections import deque
from footsies_gym.moves import FootsiesMove
from itertools import cycle
from agents.action import ActionMap
from opponents.base import Opponent, OpponentManager
from torch.utils.tensorboard import SummaryWriter # type: ignore
from os import path
from typing import Generator
from contextlib import contextmanager

LOGGER = logging.getLogger("main.curriculum")


class CurriculumManager(OpponentManager):
    def __init__(
        self,
        win_rate_threshold: float = 0.9,
        win_rate_over_episodes: int = 100,
        episode_threshold: int | None = None,
        log_dir: str | None = None,
        csv_save: bool = True,
    ):
        self._win_rate_threshold = win_rate_threshold
        self._episode_threshold = episode_threshold

        self._current_opponent_idx = 0
        self._opponents: list[CurriculumOpponent] = sorted([
            Idle(),
            Backer(),
            NSpammer(),
            BSpammer(),
            NSpecialSpammer(),
            BSpecialSpammer(),
            WhiffPunisher(),
        ], key=lambda o: o.difficulty)
        
        # An array of booleans that tracks the most recent wins of the agent (True if win, False otherwise)
        self._agent_wins: deque[bool] = deque([], maxlen=win_rate_over_episodes)
        # The total number of episodes since the last switch
        self._current_episodes: int = 0
        # How many episodes were required to beat (or skip) the opponent of the correspondent index
        self._episodes_taken_for: list[int] = []

        # This is the tracker of episodes so far, for logging
        self._episode = 0

        self._summary_writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None
        if csv_save and log_dir is not None:
            self._csv_file = open(path.join(log_dir, "performancewin_rate_against_current_curriculum_opponent.csv"), mode="wt")
            self._csv_file_writer = csv.writer(self._csv_file, "unix", quoting=csv.QUOTE_MINIMAL)
        else:
            self._csv_file = None
            self._csv_file_writer = None
    
    def _agent_surpassed_opponent(self) -> bool:
        """Check whether the agent has surpassed the current opponent enough."""
        return len(self._agent_wins) == self._agent_wins.maxlen and self.current_recent_win_rate >= self._win_rate_threshold

    def _agent_took_too_long(self) -> bool:
        """Check whether the agent took too long to beat the current opponent."""
        return self._episode_threshold is not None and self._current_episodes > self._episode_threshold

    def _advance(self) -> "CurriculumOpponent | None":
        """Advance to the next opponent. The next opponent to which this method advances is returned. If the returned opponent is `None`, then the curriculum is complete."""
        self._agent_wins.clear()
        self._current_episodes = 0
        self._current_opponent_idx = min(self._current_opponent_idx + 1, len(self._opponents))

        if self.exhausted:
            return None

        new_opponent = self.current_opponent
        return new_opponent
    
    def update_at_episode(self, game_result: float):
        if self.exhausted:
            raise ValueError("can't perform another update while the opponent manager is exhausted")

        self._agent_wins.append(game_result == 1.0)
        self._episode += 1
        self._current_episodes += 1

        opponent_change = False
        win_rate = self.current_recent_win_rate

        agent_surpassed_opponent = self._agent_surpassed_opponent()
        agent_took_too_long = self._agent_took_too_long()
        if agent_surpassed_opponent or agent_took_too_long:
            previous_opponent = self.current_opponent
            previous_wins = sum(self._agent_wins)
            previous_episodes = self._current_episodes

            self._episodes_taken_for.append(self._current_episodes)

            new_opponent = self._advance()

            if agent_surpassed_opponent:
                LOGGER.info(f"Agent has surpassed opponent {previous_opponent.__class__.__name__} with a win rate of {previous_wins / self._agent_wins.maxlen:.0%} over the recent {self._agent_wins.maxlen} episodes after {previous_episodes} episodes. Switched to {new_opponent.__class__.__name__}") # type: ignore
            elif agent_took_too_long:
                LOGGER.info(f"Agent took too long to beat {previous_opponent.__class__.__name__}, having a win rate of {previous_wins / self._agent_wins.maxlen:.0%} over the recent {self._agent_wins.maxlen} episodes after {previous_episodes} episodes. Switched to {new_opponent.__class__.__name__}") # type: ignore
            opponent_change = True

            # Reset the wins against current opponent
            self._agent_wins.clear()

        # Logging
        if self._summary_writer is not None:
            self._summary_writer.add_scalar(
                "Performance/Win rate against current curriculum opponent",
                win_rate,
                self._episode,
            )
        if self._csv_file_writer is not None:
            self._csv_file_writer.writerow((self._episode, win_rate, int(opponent_change)))

        return opponent_change

    @property
    def current_opponent(self) -> "CurriculumOpponent | None":
        return self._opponents[self._current_opponent_idx] if self._current_opponent_idx < len(self._opponents) else None

    @property
    def current_opponent_idx(self) -> int:
        return self._current_opponent_idx

    @property
    def current_opponent_episodes(self) -> int:
        return self._current_episodes

    @property
    def exhausted(self) -> bool:
        return self._current_opponent_idx >= len(self._opponents)
    
    @property
    def current_recent_win_rate(self) -> float:
        return sum(self._agent_wins) / len(self._agent_wins) if self._agent_wins else 0.5

    @property
    def episode_threshold(self) -> int | None:
        return self._episode_threshold

    def episodes_taken_for_opponent(self, opponent_idx: int) -> int:
        """The number of episodes that was required to surpass (or skip) the opponent with ID `opponent_idx`."""
        return self._episodes_taken_for[opponent_idx]

    def reset(self):
        """Reset the curriculum state to the beginning."""
        self._current_opponent_idx = 0
        self._agent_wins.clear()
        self._current_episodes = 0
        self._episodes_taken_for.clear()
        self._episode = 0

    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()


class CurriculumOpponent(Opponent):

    @abstractmethod
    def peek(self, next_obs: dict) -> torch.Tensor:
        """Return the probability distribution of the action that the agent will perform next (the next `act` call). Both observation and the probability distribution are tensors."""

    @property
    @abstractmethod
    def difficulty(self) -> int:
        """The estimated relative difficulty of this opponent."""

    def reset(self):
        """Reset any opponent state that was built during this episode."""


class CurriculumOpponentSimple(CurriculumOpponent):
    """Curriculum opponent that performs simple actions. Merely to simplify code and avoid duplication. Subclasses should call `super().__init__()` on the constructor and `super().reset()` on `reset()`."""
    def __init__(self):
        self._primitive_action_queue: deque[tuple[bool, bool, bool]] = deque([])
        self._simple_action = ActionMap.simple_from_move(FootsiesMove.STAND)

    @abstractmethod
    def compute_action(self, obs: dict) -> FootsiesMove:
        """Compute the simple action to perform at this moment"""

    def _advance_action(self, obs: dict):
        simple_action = self.compute_action(obs)
        self._simple_action = ActionMap.simple_from_move(simple_action)
        # We need to correct the action since it will be performed on the other side of the screen
        self._primitive_action_queue.extend(ActionMap.invert_primitive(p) for p in ActionMap.simple_as_move_to_primitive(simple_action))

    def act(self, obs: dict, info: dict) -> tuple[bool, bool, bool]:
        # If there were actions that was previously queued, perform them.
        # We also clean up the discrete action queue since they should be in sync.
        if self._primitive_action_queue:
            return self._primitive_action_queue.popleft()

        self._advance_action(obs)

        return self._primitive_action_queue.popleft()

    def peek(self, next_obs: dict) -> torch.Tensor:
        if self._primitive_action_queue:
            return nn.functional.one_hot(torch.tensor(self._simple_action), num_classes=ActionMap.n_simple()).float()
        
        self._advance_action(next_obs)

        return nn.functional.one_hot(torch.tensor(self._simple_action), num_classes=ActionMap.n_simple()).float()

    def reset(self):
        self._primitive_action_queue.clear()
        self._simple_action = ActionMap.simple_from_move(FootsiesMove.STAND)


class Idle(CurriculumOpponent):
    def __init__(self):
        self._probs = torch.zeros((ActionMap.n_simple(),)).float()
        self._probs[ActionMap.simple_from_move(FootsiesMove.STAND)] = 1.0

    def act(self, obs: dict, info: dict) -> tuple[bool, bool, bool]:
        return (False, False, False)
    
    def peek(self, next_obs: dict) -> torch.Tensor:
        return self._probs

    @property
    def difficulty(self) -> int:
        return 0
    

class Backer(CurriculumOpponent):
    def __init__(self):
        self._probs = torch.zeros((ActionMap.n_simple(),)).float()
        self._probs[ActionMap.simple_from_move(FootsiesMove.BACKWARD)] = 1.0

    def act(self, obs: dict, info: dict) -> tuple[bool, bool, bool]:
        return (False, True, False)

    def peek(self, next_obs: dict) -> torch.Tensor:
        return self._probs

    @property
    def difficulty(self) -> int:
        return 0


class NSpammer(CurriculumOpponentSimple):
    def __init__(self, walk_for: int = 3):
        super().__init__()
        self._walk_for = walk_for
        self._walk_for_counter = 0
    
    def compute_action(self, obs: dict) -> FootsiesMove:
        agent_move = ActionMap.move_from_move_index(obs["move"][0])
        our_move = ActionMap.move_from_move_index(obs["move"][1])
        
        self._walk_for_counter = (self._walk_for_counter - 1) % (self._walk_for + 1)
        
        # Calm down
        if our_move != FootsiesMove.FORWARD:
            return FootsiesMove.FORWARD

        # If the agent was hit, hit them! But not if they are blocking
        if agent_move == FootsiesMove.DAMAGE:
            return FootsiesMove.N_ATTACK
    
        # If walked enough, attack
        if self._walk_for_counter == 0:
            return FootsiesMove.N_ATTACK

        # Just walk forward for some time
        return FootsiesMove.FORWARD

    @property
    def difficulty(self) -> int:
        return 2


class BSpammer(CurriculumOpponentSimple):
    def __init__(self, walk_for: int = 3):
        super().__init__()
        self._walk_for = walk_for
        self._walk_for_counter = 0
    
    def compute_action(self, obs: dict) -> FootsiesMove:
        agent_move = ActionMap.move_from_move_index(obs["move"][0])
        our_move = ActionMap.move_from_move_index(obs["move"][1])
        
        self._walk_for_counter = (self._walk_for_counter - 1) % (self._walk_for + 1)
        
        # Calm down
        if our_move != FootsiesMove.FORWARD:
            return FootsiesMove.FORWARD

        # If the agent was hit, hit them! But not if they are blocking
        if agent_move == FootsiesMove.DAMAGE:
            return FootsiesMove.B_ATTACK
    
        # If walked enough, attack
        if self._walk_for_counter == 0:
            return FootsiesMove.B_ATTACK

        # Just walk forward for some time
        return FootsiesMove.FORWARD

    @property
    def difficulty(self) -> int:
        return 2


class NSpecialSpammer(CurriculumOpponent):
    def __init__(self):
        self.reset()
        self._probs = torch.zeros((ActionMap.n_simple(),)).float()
        self._probs[ActionMap.simple_from_move(FootsiesMove.N_SPECIAL)] = 1.0
    
    def act(self, obs: dict, info: dict) -> tuple[bool, bool, bool]:
        return next(self._action_cycle)
    
    def peek(self, next_obs: dict) -> torch.Tensor:
        return self._probs

    @property
    def difficulty(self) -> int:
        return 1

    def reset(self):
        self._action_cycle = cycle([
            *([(False, True, True)] * 60),
            (False, False, False),
        ])


class BSpecialSpammer(CurriculumOpponent):
    def __init__(self):
        self.reset()
        self._probs = torch.zeros((ActionMap.n_simple(),)).float()
        self._probs[ActionMap.simple_from_move(FootsiesMove.B_SPECIAL)] = 1.0
    
    def act(self, obs: dict, info: dict) -> tuple[bool, bool, bool]:
        return next(self._action_cycle)
    
    def peek(self, next_obs: dict) -> torch.Tensor:
        return self._probs

    @property
    def difficulty(self) -> int:
        return 1

    def reset(self):
        self._action_cycle = cycle([
            *([(False, True, True)] * 60),
            (True, False, False),
        ])


class WhiffPunisher(CurriculumOpponentSimple):
    def __init__(self, keepout_distance: float = 2.624):
        super().__init__()
        self._keepout_distance = keepout_distance

    def compute_action(self, obs: dict) -> FootsiesMove:
        agent_move = ActionMap.move_from_move_index(obs["move"][0])
        agent_move_frame = obs["move_frame"][0]
        agent_position = obs["position"][0]
        our_position = obs["position"][1]
        
        # If we hit the agent, finish them off
        if agent_move == FootsiesMove.DAMAGE:
            return FootsiesMove.N_ATTACK

        # Move towards the agent by default if distance between players is too great
        distance_between_players = our_position - agent_position
        if distance_between_players > self._keepout_distance:
            return FootsiesMove.FORWARD

        if agent_move in (FootsiesMove.N_ATTACK, FootsiesMove.B_ATTACK, FootsiesMove.N_SPECIAL, FootsiesMove.B_SPECIAL):
            # If the agent is in recovery and we can hit them before they recover...
            if agent_move.in_recovery(agent_move_frame) and agent_move_frame + FootsiesMove.N_ATTACK.value.startup < agent_move.value.duration:
                # ... punish them
                return FootsiesMove.N_ATTACK

            # If the agent is beginning to perform the move...
            else:
                # ... just block
                return FootsiesMove.BACKWARD
        
        # Don't get too close to the opponent
        return FootsiesMove.BACKWARD

    @property
    def difficulty(self):
        return 3
