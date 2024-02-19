import torch
from typing import Any, Callable
from collections import namedtuple
from dataclasses import dataclass, field


RPSObservation = tuple | torch.Tensor


# Actions
#  0: rock
#  1: paper
#  2: scissors
# TODO: test
# Temporally extended actions
#  0: .         2 [attack]   ..     5 steps
#  1: ..          [throw]    ...    6 steps
#  2:           3 [dodge]    .      4 steps
#   
#   attack wins against throw
#   throw wins against dodge
#   dodge wins against attack

@dataclass
class TemporalAction:
    name:           str
    startup:        int
    active:         int
    recovery:       int
    wins_against:   set[str] = field(default_factory=set)
    current_step:   int = 0

    def terminated(self) -> bool:
        return self.current_step >= self.startup + self.active + self.recovery

    def active(self) -> bool:
        return self.startup <= self.current_step < self.startup + self.active

    def advance(self):
        self.current_step += 1
    
    @classmethod
    def resolve(cls, p1: "TemporalAction", p2: "TemporalAction") -> int:
        if p1.active() and p2.active():
            if p1.name in p2.wins_against:
                return 1
            elif p2.name in p1.wins_against:
                return -1
            return 0
        elif p1.active():
            return 1
        elif p2.active():
            return -1
        return 0


DEFAULT_TEMPORAL_ACTIONS: list[TemporalAction] = [
    TemporalAction("rock", 1, 1, 4, {"scissors"}),
    TemporalAction("paper", 2, 1, 3, {"rock"}),
    TemporalAction("scissors", 4, 2, 1, {"paper"}),
    TemporalAction("guard", 0, 1, 0, {"rock", "paper", "scissors"}),
    TemporalAction("break", 0, 1, 3, {"guard", "rock", "paper", "scissors"}),
]


class RPS:
    def __init__(
        self,
        opponent: Callable[[RPSObservation], int],
        dense_reward: bool = False,
        health: int = 2, # a health of 2 is equivalent to a best-of-3 match
        flattened: bool = True,
        use_temporal_actions: bool = False,
        observation_include_play: bool = False,
        observation_include_move_progress: bool = False,
        observation_transformer: Callable[[RPSObservation], Any] = lambda o: o,
        temporal_actions: list[TemporalAction] = DEFAULT_TEMPORAL_ACTIONS,
    ):
        self.opponent = opponent
        self.dense_reward = dense_reward
        self.health = health
        self.flattened = flattened
        self.use_temporal_actions = use_temporal_actions
        self.observation_include_play = observation_include_play
        self.observation_include_move_progress = observation_include_move_progress
        self.transform_observation = observation_transformer
        self.temporal_actions = temporal_actions

        self.current_observation = 0
        self.p1_health = 0
        self.p2_health = 0
        # For temporal actions only
        self.p1_current_temporal_action: TemporalAction = None
        self.p2_current_temporal_action: TemporalAction = None

        self.current_step = 0
        # For calculating dense reward
        self._cumulative_episode_reward = 0
    
    def craft_observation(
        self,
        p1_health: int,
        p2_health: int,
        p1_play: int = None,
        p2_play: int = None,
    ):
        if self.flattened:
            if p1_health <= 0 or p2_health <= 0:
                return None
            
            parts = [
                torch.nn.functional.one_hot(torch.tensor(p1_health - 1), num_classes=self.health),
                torch.nn.functional.one_hot(torch.tensor(p2_health - 1), num_classes=self.health),
            ]
            
            if self.observation_include_play:
                parts.append(torch.nn.functional.one_hot(torch.tensor(p1_play), num_classes=self.play_dim))
                parts.append(torch.nn.functional.one_hot(torch.tensor(p2_play), num_classes=self.play_dim))
            
            return self.transform_observation(torch.hstack(parts))
        
        parts = [
            p1_health,
            p2_health,
        ]

        if self.observation_include_play:
            parts.append(p1_play)
            parts.append(p2_play)

        return self.transform_observation(tuple(parts))

    def observation(self, p1_play: int, p2_play: int) -> RPSObservation:
        return self.craft_observation(self.p1_health, self.p2_health, p1_play, p2_play)

    def info(self, p1_play: int, p2_play: int, p1_action: int, p2_action: int) -> dict:
        return {
            "p1_health": self.p1_health,
            "p2_health": self.p2_health,
            "p1_play": p1_play,
            "p2_play": p2_play,
            "p1_action": p1_action,
            "p2_action": p2_action,
            "step": self.current_step,
        }

    def reset(self) -> tuple[RPSObservation, dict]:
        self.p1_health = self.health
        self.p2_health = self.health
        self.current_step = 0
        self._cumulative_episode_reward = 0

        # Assume initial play by both players is 0 (rock)
        self.current_observation = self.observation(0, 0)
        self.current_info = self.info(0, 0, 0, 0)
        return self.current_observation, self.current_info

    def reward(self, result: int, terminated: bool) -> float:
        if self.dense_reward:
            reward = 0.0
            reward += result / self.health
            
            self._cumulative_episode_reward += reward

            if terminated:
                reward += result - self._cumulative_episode_reward
            
            return reward
        
        if terminated:
            if self.p1_health > self.p2_health:
                return 1.0
            elif self.p1_health < self.p2_health:
                return -1.0
            else:
                return 0.0

        return 0.0

    def resolve(self, p1_action: int, p2_action: int) -> int:
        if self.use_temporal_actions:
            p1_temporal_action = self.temporal_actions[p1_action]()
            p2_temporal_action = self.temporal_actions[p2_action]()

            # If we are using temporal actions, then we need to keep track of the performed actions, and ignore those that were made while another was still being performed
            if self.p1_current_temporal_action is None:
                self.p1_current_temporal_action = p1_temporal_action
            if self.p2_current_temporal_action is None:
                self.p2_current_temporal_action = p2_temporal_action
            
            TemporalAction.resolve(self.p1_current_temporal_action, self.p2_current_temporal_action)

            p1_temporal_action.advance()
            p2_temporal_action.advance()

            if p1_temporal_action.terminated():
                self.p1_current_temporal_action = None
            if p2_temporal_action.terminated():
                self.p2_current_temporal_action = None

        else:
            offset = 1 - p1_action
            reward = ((p1_action + offset) % 3) - ((p2_action + offset) % 3)
            return reward

    # NOTE: we decouple 'action' and 'play': 'action' is what the agents do on the environment, the 'play' is the observed result of that action.
    #       This might be useful for introducing complexity in the future
    def step(self, action: int) -> tuple[RPSObservation, float, bool, bool, dict]:
        p1_action = action
        p2_action = self.opponent(self.current_observation, self.current_info)
        
        result = self.resolve(p1_action, p2_action)

        p1_play = p1_action
        p2_play = p2_action

        if result > 0:
            self.p2_health -= 1
        elif result < 0:
            self.p1_health -= 1
        else:
            self.p1_health -= 1
            self.p2_health -= 1

        terminated = self.p1_health <= 0 or self.p2_health <= 0

        reward = self.reward(result, terminated)

        self.current_step += 1
        self.current_observation = self.observation(p1_play, p2_play)
        self.current_info = self.info(p1_play, p2_play, p1_action, p2_action)
        return self.current_observation, reward, terminated, False, self.current_info

    def set_opponent(self, opponent: Callable[[RPSObservation], int]):
        self.opponent = opponent

    @property
    def play_dim(self) -> int:
        return 3

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def observation_dim(self) -> int:
        if self.flattened:
            return 2 * self.health + 2 * self.play_dim * self.observation_include_play
        return 2 + 2 * self.observation_include_play
    

class RPSTemporalActionResolver:
    def __init__(
        self,
        temporal_actions: dict[str, TemporalAction] = DEFAULT_TEMPORAL_ACTIONS,
    ):
        self.temporal_actions = temporal_actions

        self.current_p1_action = None
        self.current_p2_action = None

    def p1_action(self, action: int):
        self.current_p1_action = action
        return action
    
    def p2_action(self, action: int):
        self.current_p2_action = action
        return action

    def action_dim(self) -> int:
        return len(self.temporal_actions)
