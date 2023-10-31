from collections import deque, namedtuple
from typing import Iterator

import cv2 as cv
import numpy as np
import torch
from agents.base import FootsiesAgentBase
from torch import nn
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatten_space


# Convolutional
class QNetwork(nn.Module):
    def __init__(self, n_actions: int):
        self.layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(8, 8), stride=4, padding="valid"), # expects input of size (N, C, W, H). N is the number of batches, and here C is 1 (grayscale so 1 channel)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=2, padding="valid"),
            nn.ReLU(),
            nn.Linear(..., 256),
            nn.Linear(256, n_actions),
        )

    def forward(self):
        ...


Transition = namedtuple("Transition", ["obs", "action", "next_obs", "reward"])


class ReplayMemory:
    def __init__(self, capacity: int):
        self.deque = deque([], maxlen=capacity)

    def add_transition(self, transition: Transition):
        self.deque.append(transition)

    def __iter__(self) -> Iterator[Transition]:
        return self.deque.__iter__()

    def random_sample(self) -> Iterator[Transition]:
        for transition in self.deque:
            if np.random.random() < 0.5:
                yield transition


class FootsiesAgent(FootsiesAgentBase):
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 replay_memory_capacity: int):
        # For random sampling when exploring
        self.action_space = action_space

        # Observations assumed to be flattened
        n_observations = flatten_space(observation_space).shape[0]
        n_actions = flatten_space(action_space).shape[0]
        self.q_network = QNetwork(n_observations, n_actions)
        self.loss_function = nn.MSELoss()

        self.replay_memory = ReplayMemory(replay_memory_capacity)
        # For frame stacking
        self.history = deque([], maxlen=4)
    
    def preprocess_history(self) -> "any":
        # Don't preprocess history if it's not full yet
        if len(self.history) < 4:
            return None
        
        frames = []
        for frame in self.history:
            # Convert to gray-scale
            gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            # Down-sample from 210x160 to 110x84
            downsampled = cv.resize(gray, (110, 84), interpolation=cv.INTER_LINEAR)
            # Crop to centered 84x84 (y and x are different)
            cropped = downsampled[:, 13:-13]
            
            cropped: np.ndarray
            frames.append(cropped.flatten().reshape((-1, 1)))

        # NOTE: I'm assuming the order of the frames when stacking doesn't impact anaything
        return np.stack(frames, axis=1)

    # epsilon-greedy
    def act(self, obs) -> "any":
        if np.random.random() < self.epsilon:
            return self.action_space.sample()

        


    def update(self, next_obs, reward: float, terminated: bool, truncated: bool):
        ...

    def load(self, folder_path: str):
        ...

    def save(self, folder_path: str):
        ...
