import torch
from typing import Any, Callable


RPSObservation = tuple | torch.Tensor


# Actions
#  0: rock
#  1: paper
#  2: scissors
class RPS:
    def __init__(
        self,
        opponent: Callable[[RPSObservation], int],
        dense_reward: bool = False,
        health: int = 2, # a health of 2 is equivalent to a best-of-3 match
        flattened: bool = True,
        observation_include_play: bool = False,
        observation_transformer: Callable[[RPSObservation], Any] = lambda o: o,
    ):
        self.opponent = opponent
        self.dense_reward = dense_reward
        self.health = health
        self.flattened = flattened
        self.observation_include_play = observation_include_play
        self.transform_observation = observation_transformer

        self.current_observation = 0
        self.p1_health = 0
        self.p2_health = 0

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

    @staticmethod
    def resolve(p1_action: int, p2_action: int) -> int:
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