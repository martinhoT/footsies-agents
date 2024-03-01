import torch
from typing import Any, Callable, List
from dataclasses import dataclass


RPSObservation = tuple | torch.Tensor


# Actions
#  0: rock
#  1: paper
#  2: scissors
#
# Temporally extended actions
#  0: [wait]                        1 step
#  1: .         2 [attack]   ..     5 steps
#  2: ..          [throw]    ...    6 steps
#  3:           3 [dodge]    .      4 steps
#   
#   attack wins against throw
#   throw wins against dodge
#   dodge wins against attack

@dataclass
class TemporalAction:
    startup:        int
    active:         int
    recovery:       int
    current_step:   int = 0

    @property
    def terminated(self) -> bool:
        return self.current_step >= self.duration

    @property
    def is_active(self) -> bool:
        return self.startup <= self.current_step < self.startup + self.active

    def advance(self):
        self.current_step += 1

    def terminate(self):
        self.current_step = self.duration
    
    @property
    def duration(self):
        return self.startup + self.active + self.recovery
    

class WaitTA(TemporalAction):
    pass

class AttackTA(TemporalAction):
    pass

class ThrowTA(TemporalAction):
    pass

class DodgeTA(TemporalAction):
    pass


def resolve_temporal_interaction(p1: TemporalAction, p2: TemporalAction, log: bool = False) -> tuple[int, int]:
    """Resolve the temporal actions of two players, returning the reward in player 1's perspective."""
    reward = (0, 0)
    interaction = False

    if p1.is_active and p2.is_active:
        if isinstance(p1, ThrowTA) and isinstance(p2, ThrowTA):
            if log: print("Throw clash!")
            reward = 0, 0
            interaction = True
        elif isinstance(p1, AttackTA) and isinstance(p2, AttackTA):
            if log: print("Attack clash!")
            reward = -1, -1
            interaction = True
        elif isinstance(p1, AttackTA):
            if isinstance(p2, DodgeTA):
                if log: print("P2 dodged P1's attack!")
            else:
                if log: print("P1's attack wins")
                reward = 1, -1
                interaction = True
        elif isinstance(p2, AttackTA):
            if isinstance(p1, DodgeTA):
                if log: print("P1 dodged P2's attack!")
            else:
                if log: print("P2's attack wins")
                reward = -1, 1
                interaction = True
        elif isinstance(p1, ThrowTA):
            if log: print("P1 threw P2!")
            reward = 1, -1
        elif isinstance(p2, ThrowTA):
            if log: print("P2 threw P1!")
            reward = -1, 1

    elif p1.is_active:
        if isinstance(p1, ThrowTA) or isinstance(p1, AttackTA):
            if log: print("P1 damaged P2!")
            reward = 1, -1
            interaction = True

    elif p2.is_active:
        if isinstance(p2, ThrowTA) or isinstance(p2, AttackTA):
            if log: print("P2 damaged P1!")
            reward = -1, 1
            interaction = True

    # In case an interaction occurred, we interrupt both actions and begin from neutral again
    if interaction:
        p1.terminate()
        p2.terminate()
    
    return reward


DEFAULT_TEMPORAL_ACTIONS = [
    lambda: WaitTA(0, 0, 1),
    lambda: AttackTA(1, 2, 2),
    lambda: ThrowTA(2, 1, 3),
    lambda: DodgeTA(0, 3, 1),
]


class RPS:
    def __init__(
        self,
        opponent: Callable[[RPSObservation], int],
        dense_reward: bool = False,
        health: int = 2,
        flattened: bool = True,
        use_temporal_actions: bool = False,
        observation_include_move: bool = False,
        observation_include_move_progress: bool = False,
        observation_transformer: Callable[[RPSObservation], Any] = lambda o: o,
        temporal_actions: list[Callable[[], TemporalAction]] = DEFAULT_TEMPORAL_ACTIONS,
        time_limit: int = 100000,
        time_limit_as_truncation: bool = False,
    ):
        """
        Environment implementing the rock-paper-scissors (RPS) game.
        Multiple features can be added, to progressively build complexity akin to that of fighting games.

        Parameters
        ----------
        - `opponent`: which opponent to use
        - `dense_reward`: whether to use a dense reward scheme (reward on every hit)
        - `health`: the health of the players. For instance, a health of 2 is equivalent to a best-of-3 RPS match
        - `flattened`: whether to return a flattened observation
        - `use_temporal_actions`: whether to use temporally extended actions. This incorporates a completely different action set
        - `observation_include_play`: whether to include each players' play/move in the observation. Note that this is distinct from the action. A play is the perceptual effect of an action
        - `observation_include_move_progress`: whether to include the move progress in the observation
        - `observation_transformer`: a function to transform the observation before returning it
        - `temporal_actions`: the temporally extended action set to use. Recommended to use the default
        - `time_limit`: the maximum number of steps per episode
        - `time_limit_as_truncation`: whether to treat the time limit as episode termination or truncation. Fighting games treat the time limit as episode termination
        """
        if not use_temporal_actions and observation_include_move_progress:
            raise ValueError("invalid arguments, asked for inclusion of the move progress of temporal actions without using temporal actions")
        
        self.opponent = opponent
        self.dense_reward = dense_reward
        self.health = health
        self.flattened = flattened
        self.use_temporal_actions = use_temporal_actions
        self.observation_include_move = observation_include_move
        self.observation_include_move_progress = observation_include_move_progress
        self.transform_observation = observation_transformer
        self.temporal_actions = temporal_actions
        self.time_limit = time_limit
        self.time_limit_as_truncation = time_limit_as_truncation

        self.current_observation = 0
        self.p1_health = 0
        self.p2_health = 0
        # For temporal actions only
        self.p1_current_temporal_action: TemporalAction = None
        self.p2_current_temporal_action: TemporalAction = None
        self.temporal_action_idx_map: List[type[TemporalAction]] = [type(t()) for t in temporal_actions]

        self.current_step = 0
        # For calculating dense reward
        self._cumulative_episode_reward = 0
    
    def craft_observation(
        self,
        p1_health: int,
        p2_health: int,
        p1_move: int = None,
        p2_move: int = None,
        p1_move_progress: float = 0.0,
        p2_move_progress: float = 0.0,
    ):
        if self.flattened:
            if p1_health <= 0 or p2_health <= 0:
                return None
            
            parts = [
                torch.nn.functional.one_hot(torch.tensor(p1_health - 1), num_classes=self.health),
                torch.nn.functional.one_hot(torch.tensor(p2_health - 1), num_classes=self.health),
            ]
            
            if self.observation_include_move:
                parts.append(torch.nn.functional.one_hot(torch.tensor(p1_move), num_classes=self.move_dim))
                parts.append(torch.nn.functional.one_hot(torch.tensor(p2_move), num_classes=self.move_dim))
            
            if self.observation_include_move_progress:
                parts.append(torch.tensor(p1_move_progress))
                parts.append(torch.tensor(p2_move_progress))

            return self.transform_observation(torch.hstack(parts))
        
        parts = [
            p1_health,
            p2_health,
        ]

        if self.observation_include_move:
            parts.append(p1_move)
            parts.append(p2_move)
        
        if self.observation_include_move_progress:
            parts.append(torch.tensor(p1_move_progress))
            parts.append(torch.tensor(p2_move_progress))

        return self.transform_observation(tuple(parts))

    def observation(self, p1_move: int, p2_move: int) -> RPSObservation:
        p1_normalized_duration = (self.p1_current_temporal_action.current_step / self.p1_current_temporal_action.duration) if self.p1_current_temporal_action is not None else 0.0
        p2_normalized_duration = (self.p2_current_temporal_action.current_step / self.p2_current_temporal_action.duration) if self.p2_current_temporal_action is not None else 0.0

        return self.craft_observation(
            self.p1_health, self.p2_health,
            p1_move, p2_move,
            p1_normalized_duration, p2_normalized_duration,
        )

    def info(self, p1_move: int, p2_move: int, p1_action: int, p2_action: int) -> dict:
        return {
            "p1_health": self.p1_health,
            "p2_health": self.p2_health,
            "p1_move": p1_move,
            "p2_move": p2_move,
            "p1_action": p1_action,
            "p2_action": p2_action,
            "p1_frame": None if self.p1_current_temporal_action is None else self.p1_current_temporal_action.current_step,
            "p2_frame": None if self.p2_current_temporal_action is None else self.p2_current_temporal_action.current_step,
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

    def reward(self, result: int, game_over: bool) -> float:
        if self.dense_reward:
            reward = 0.0
            reward += result / self.health
            
            self._cumulative_episode_reward += reward

            if game_over:
                reward += result - self._cumulative_episode_reward
            
            return reward
        
        if game_over:
            if self.p1_health > self.p2_health:
                return 1.0
            elif self.p1_health < self.p2_health:
                return -1.0
            else:
                return 0.0

        return 0.0

    def resolve(self, p1_action: int, p2_action: int) -> int:
        if self.use_temporal_actions:
            # If we are using temporal actions, then we need to keep track of the performed actions, and ignore those that were made while another was still being performed
            if self.p1_current_temporal_action is None:
                self.p1_current_temporal_action_idx = p1_action
                self.p1_current_temporal_action = self.temporal_actions[p1_action]()
            if self.p2_current_temporal_action is None:
                self.p2_current_temporal_action_idx = p2_action
                self.p2_current_temporal_action = self.temporal_actions[p2_action]()
            
            p1_result, p2_result = resolve_temporal_interaction(self.p1_current_temporal_action, self.p2_current_temporal_action)

            self.p1_current_temporal_action.advance()
            self.p2_current_temporal_action.advance()

            if self.p1_current_temporal_action.terminated:
                self.p1_current_temporal_action = None
            if self.p2_current_temporal_action.terminated:
                self.p2_current_temporal_action = None
            
            return p1_result, p2_result

        else:
            offset = 1 - p1_action
            reward = ((p1_action + offset) % 3) - ((p2_action + offset) % 3)
            
            if reward == 0:
                # Both players lose health when drawing the same action
                return -1, -1
            
            return reward, -reward

    # NOTE: we decouple 'action' and 'play': 'action' is what the agents do on the environment, the 'play' is the observed result of that action.
    #       This might be useful for introducing complexity in the future
    def step(self, action: int) -> tuple[RPSObservation, float, bool, bool, dict]:
        p1_action = action
        p2_action = self.opponent(self.current_observation, self.current_info)
        
        p1_result, p2_result = self.resolve(p1_action, p2_action)

        p1_move = p1_action if self.p1_current_temporal_action is None else self.temporal_action_idx(self.p1_current_temporal_action)
        p2_move = p2_action if self.p2_current_temporal_action is None else self.temporal_action_idx(self.p2_current_temporal_action)

        if p1_result < 0:
            self.p1_health -= 1
        if p2_result < 0:
            self.p2_health -= 1

        self.current_step += 1
        self.current_observation = self.observation(p1_move, p2_move)
        self.current_info = self.info(p1_move, p2_move, p1_action, p2_action)

        terminated = self.p1_health <= 0 or self.p2_health <= 0
        truncated = self.current_step >= self.time_limit
        reward = self.reward(p1_result, terminated or truncated)

        return self.current_observation, reward, terminated, truncated, self.current_info

    def set_opponent(self, opponent: Callable[[RPSObservation], int]):
        self.opponent = opponent

    def temporal_action_idx(self, temporal_action: TemporalAction) -> int:
        return self.temporal_action_idx_map.index(type(temporal_action))

    @property
    def move_dim(self) -> int:
        if self.use_temporal_actions:
            return len(self.temporal_actions)
        return 3

    @property
    def action_dim(self) -> int:
        if self.use_temporal_actions:
            return len(self.temporal_actions)
        return 3

    @property
    def observation_dim(self) -> int:
        if self.flattened:
            return 2 * self.health + 2 * self.move_dim * self.observation_include_move + 2 * self.observation_include_move_progress
        return 2 + 2 * self.observation_include_move + 2 * self.observation_include_move_progress
    