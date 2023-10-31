from collections import deque, namedtuple
import random
from typing import Iterator, List

import cv2 as cv
import numpy as np
import torch
import os
from agents.base import FootsiesAgentBase
from torch import nn
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatten_space
from torch.utils.tensorboard import SummaryWriter


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

    def forward(self, x):
        return self.layers(x)


Transition = namedtuple("Transition", ["obs", "action", "next_obs", "reward", "terminated"])


class ReplayMemory:
    def __init__(self, capacity: int):
        self.deque = deque([], maxlen=capacity)

    def add_transition(self, transition: Transition):
        self.deque.append(transition)

    def __iter__(self) -> Iterator[Transition]:
        return self.deque.__iter__()

    def random_sample(self, batch_size) -> List[Transition]:
        return random.sample(self.deque, batch_size)


# As implemented in: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
class FootsiesAgent(FootsiesAgentBase):
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 replay_memory_capacity: int,
                 replay_memory_batch_size: int = 32,
                 discount_factor: float = ...,
                 learning_rate: float = ...,
                 epsilon: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay_factor: float = (1.0 - 0.1) / 1_000_000):
        # For random sampling when exploring
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay_factor = epsilon_decay_factor

        # Observations assumed to be flattened
        n_observations = flatten_space(observation_space).shape[0]
        n_actions = flatten_space(action_space).shape[0]
        self.q_network = QNetwork(n_observations, n_actions)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.q_network.parameters(), lr=learning_rate)

        self.replay_memory = ReplayMemory(replay_memory_capacity)
        self.replay_memory_batch_size = replay_memory_batch_size
        # For frame stacking
        self.history = deque([], maxlen=4)

        self.current_stack = None
        self.chosen_action = None

        self.summary_writer = SummaryWriter()
        self.current_step = 0
        self.test_stacks = None # This is an array of states on which the Q-network will be evaluated
    
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

        # NOTE: I'm assuming the order of the frames when stacking doesn't impact anything
        return np.stack(frames, axis=1)

    # Epsilon-greedy
    def act(self, obs) -> "any":
        self.history.append(obs)
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        
        stack = self.preprocess_history()
        if stack is None:
            return self.action_space.sample()
        
        action = torch.argmax(self.q_network(stack)).item()

        self.current_stack = stack
        self.chosen_action = action

        return action

    def update(self, next_obs, reward: float, terminated: bool, truncated: bool):
        self.history.append(next_obs)
        next_stack = self.preprocess_history()

        transition = Transition(self.current_stack, self.chosen_action, next_stack, reward, terminated)
        self.replay_memory.add_transition(transition)

        batch = self.replay_memory.random_sample(self.replay_memory_batch_size)

        stacks = torch.tensor([transition.obs for transition in batch])
        target = torch.tensor([transition.reward if transition.terminated else (transition.reward + self.discount_factor * torch.max(self.q_network(transition.next_obs)))])

        self.optimizer.zero_grad()
        loss = self.loss_function(self.q_network(stacks), target)
        loss.backward()
        self.optimizer.step()

        self.current_step += 1
        self.summary_writer.add_scalar("Average Q-value", self.evaluate_q_network())

        # No history between episodes ofc
        if terminated or truncated:
            self.history.clear()

    def collect_test_states(self, env, test_set_size: int):
        state, _ = env.reset()

        self.history.append(state)
        test_stacks = set()
        for _ in range(test_set_size):
            state, _, terminated, truncated, _ = env.step()
            self.history.append(state)
            stack = self.preprocess_history()
            if stack is not None:
                test_stacks.add(stack)

            if terminated or truncated:
                self.history.clear()
        
        self.test_stacks = np.array(test_stacks)
        self.history.clear() # don't influence training

    def evaluate_q_network(self):
        # Average maximum Q-values for each state (makes sense to max since we use a greedy policy)
        with torch.no_grad():
            return torch.mean(torch.max(self.q_network(self.test_stacks)))

    def load(self, folder_path: str):
        model_path = os.path.join(folder_path, "model_weights.pth")
        self.q_network.load_state_dict(torch.load(model_path))

    def save(self, folder_path: str):
        model_path = os.path.join(folder_path, "model_weights.pth")
        torch.save(self.q_network.state_dict(), model_path)
