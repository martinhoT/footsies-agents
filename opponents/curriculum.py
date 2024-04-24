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
            # UnsafePunisher(), # The UnsafePunisher can easily stall the agent. At least WhiffPunisher approaches the agent, but stalling should be possible there as well.
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
            self._csv_file = open(path.join(log_dir, "performancewin_rate_against_current_curriculum_opponent"))
            self._csv_file_writer = csv.writer(
                csvfile=self._csv_file,
                dialect="unix",
                quoting=csv.QUOTE_MINIMAL,
            )
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
            self._csv_file_writer.writerow((self._episode, win_rate))

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
        if self._csv_file_writer is not None:
            self._csv_file_writer


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


class NSpammer(CurriculumOpponent):
    def __init__(self):
        self.reset()

    def act(self, obs: dict, info: dict) -> tuple[bool, bool, bool]:
        if self._peeked_action is not None:
            action = self._peeked_action
            self._peeked_action = None
            return action
        
        return next(self._action_cycle)
    
    def peek(self, next_obs: dict) -> torch.Tensor:
        if self._peeked_action is None:
            self._peeked_action = next(self._action_cycle)

        simple = FootsiesMove.STAND
        if self._peeked_action == (False, False, True):
            simple = FootsiesMove.N_ATTACK
        elif self._peeked_action == (True, False, False):
            simple = FootsiesMove.FORWARD

        simple = ActionMap.simple_from_move(simple)
        return nn.functional.one_hot(torch.tensor(simple), num_classes=ActionMap.n_simple()).float()

    def reset(self):
        self._action_cycle = cycle([
            (False, False, True),
            *([(False, False, False)] * FootsiesMove.N_ATTACK.value.startup), # Wait for the attack to start before attacking again
            *([(False, False, True)] * FootsiesMove.N_ATTACK.value.active), # Attack during active frames
            *([(False, False, False)] * FootsiesMove.N_ATTACK.value.recovery), # Wait until the attack finishes before moving forward
            (True, False, False),
            (True, False, False),
            (True, False, False),
        ])
        self._peeked_action = None

    @property
    def difficulty(self) -> int:
        return 1


class BSpammer(CurriculumOpponent):
    def __init__(self):
        self.reset()
    
    def act(self, obs: dict, info: dict) -> tuple[bool, bool, bool]:
        if self._peeked_action is not None:
            action = self._peeked_action
            self._peeked_action = None
            return action
        
        return next(self._action_cycle)

    def peek(self, next_obs: dict) -> torch.Tensor:
        if self._peeked_action is None:
            self._peeked_action = next(self._action_cycle)
        
        simple = FootsiesMove.STAND
        if self._peeked_action == (True, False, True):
            simple = FootsiesMove.B_ATTACK
        elif self._peeked_action == (True, False, False):
            simple = FootsiesMove.FORWARD

        simple = ActionMap.simple_from_move(simple)
        return nn.functional.one_hot(torch.tensor(simple), num_classes=ActionMap.n_simple()).float()

    @property
    def difficulty(self) -> int:
        return 1
    
    def reset(self):
        self._action_cycle = cycle([
            (True, False, True),
            *([(False, False, False)] * FootsiesMove.B_ATTACK.value.startup), # Wait for the attack to start before attacking again
            *([(False, True, True)] * FootsiesMove.B_ATTACK.value.active), # Attack during active frames
            *([(False, False, False)] * FootsiesMove.B_ATTACK.value.recovery), # Wait until the attack finishes before moving forward
            (True, False, False),
            (True, False, False),
            (True, False, False),
        ])
        self._peeked_action = None


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
        return 2

    def reset(self):
        self._action_cycle = cycle([
            *([(False, False, True)] * 60),
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
        return 2

    def reset(self):
        self._action_cycle = cycle([
            *([(False, False, True)] * 60),
            (True, False, False),
        ])


class WhiffPunisher(CurriculumOpponent):
    def __init__(self, keepout_distance: float = 2.624):
        self._keepout_distance = keepout_distance
        self._primitive_action_queue = deque([])
        self._simple_action = FootsiesMove.STAND

    def _compute_action(self, agent_move: FootsiesMove, agent_move_frame: int, agent_position: float, our_position: float) -> FootsiesMove:
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

    def _advance_action(self, obs: dict):
        agent_move = ActionMap.move_from_move_index(obs["move"][0])

        simple_action = self._compute_action(
            agent_move=agent_move,
            agent_move_frame=obs["move_frame"][0],
            agent_position=obs["position"][0],
            our_position=obs["position"][1],
        )
        
        self._simple_action = ActionMap.simple_from_move(simple_action)
        # We need to correct the action since it will be performed on the other side of the screen
        self._primitive_action_queue.extend(ActionMap.invert_primitive(p) for p in ActionMap.simple_as_move_to_primitive(simple_action))

    # NOTE: keep in mind that the opponent has actions inverted, so FORWARD -> BACKWARD and vice versa
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

    @property
    def difficulty(self):
        return 2

    def reset(self):
        self._primitive_action_queue.clear()
        self._simple_action = FootsiesMove.STAND
    

class UnsafePunisher(CurriculumOpponent):
    """This opponent stays idle, unless the agent attacks, in which case it blocks. It will retaliate if the agent performs either `N_SPECIAL` or `B_SPECIAL` and it blocks."""
    def __init__(self):
        self._primitive_action_queue = deque([])
        self._simple_action = FootsiesMove.STAND
        self._will_punish = False

    def _compute_action(self, agent_move: FootsiesMove, our_move: FootsiesMove) -> FootsiesMove:
        # If we hit the agent, finish them off
        if agent_move == FootsiesMove.DAMAGE:
            return FootsiesMove.N_ATTACK

        # If we should start a punish, do it
        if self._will_punish:
            if our_move == FootsiesMove.STAND:
                self._will_punish = False
                return FootsiesMove.N_ATTACK
            
            # Just wait until we can act to perform the punish
            else:
                return FootsiesMove.STAND

        # If the agent has blocked, set to look for the opportunity to punish as soon as possible, and wait until then
        if agent_move in (FootsiesMove.N_SPECIAL, FootsiesMove.B_SPECIAL) and our_move in ActionMap.HIT_GUARD_STATES:
            self._will_punish = True
            return FootsiesMove.STAND
        
        # If the agent will perform an attack, block
        if agent_move in (FootsiesMove.N_ATTACK, FootsiesMove.B_ATTACK, FootsiesMove.N_SPECIAL, FootsiesMove.B_SPECIAL):
            return FootsiesMove.BACKWARD
        
        # Stand idle by default
        return FootsiesMove.STAND

    def _advance_action(self, obs: dict):
        agent_move = ActionMap.move_from_move_index(obs["move"][0])
        our_move = ActionMap.move_from_move_index(obs["move"][1])   

        simple_action = self._compute_action(agent_move, our_move)

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

    @property
    def difficulty(self) -> int:
        return 3

    def reset(self):
        self._primitive_action_queue.clear()
        self._simple_action = FootsiesMove.STAND
        self._will_punish = False
