import os

from gymnasium import Env
from agents.base import FootsiesAgentBase
from typing import Any, List, Tuple
import numpy as np
import torch
from torch import nn
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatten_space


class LiteralQNetwork(nn.Module):
    """This network predicts the Q-value of a state-action pair. Since the reward is either -1 or 1, the final activation layer is Tanh()"""

    def __init__(self, n_observations: int, n_actions: int, shallow: bool = True):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = (
            nn.Sequential(
                nn.Linear(n_observations + n_actions, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Tanh(),
            )
            if not shallow
            else nn.Sequential(
                nn.Linear(n_observations + n_actions, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Tanh(),
            )
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits


class FootsiesAgent(FootsiesAgentBase):
    """Only includes the fine-tuning phase of the Brisket method"""

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        alpha: float = 0.5,
        learning_rate: float = 0.00001,
        discount_factor: float = 0.95,
        epsilon: float = 0.95,
        epsilon_decay_rate: float = 0.0001,
        min_epsilon: float = 0.05,
        shallow: bool = True,
        device: torch.device = "cpu",
        **kwargs,
    ):
        self.action_space = action_space
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.device = device

        self.observations_length = flatten_space(observation_space).shape[0]
        self.actions_length = flatten_space(action_space).shape[0]
        self.q_network = LiteralQNetwork(
            self.observations_length, self.actions_length, shallow=shallow
        )

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.learning_rate
        )
        self.loss_function = nn.MSELoss()

        # Accumulate training data (only learn after each game)
        self.trainX = torch.tensor([], device=self.device, requires_grad=False)
        self.trainY = torch.tensor([], device=self.device, requires_grad=False)

        self.current_iteration = 0
        self.current_observation = None
        self.current_action = None

        # For evaluation
        self._test_states = None

    def act(self, obs) -> Any:
        if not isinstance(obs, torch.Tensor):
            obs = (
                torch.tensor(obs, dtype=torch.float32).to(self.device).reshape((1, -1))
            )

        self.current_observation = obs
        action, _ = self.policy(obs)
        self.current_action = action
        return action

    def policy(self, obs) -> Tuple[int, float]:
        if np.random.random() < self.epsilon:
            random_action = self.action_space.sample()
            random_action_oh = self.action_oh(random_action)
            return random_action, self.q_value(obs, random_action_oh)

        q_values = [
            self.q_value(obs, self.action_oh(action))
            for action in range(self.actions_length)
        ]
        return np.argmax(q_values), np.max(q_values)

    def q_value(self, obs, action_oh) -> float:
        with torch.no_grad():
            return self.q_network(torch.cat((obs, action_oh), dim=1)).item()

    def action_oh(self, action: int):
        action_one_hot = np.zeros((1, self.actions_length))
        action_one_hot[0, action] = 1
        return torch.tensor(
            action_one_hot, dtype=torch.float32, device=self.device, requires_grad=False
        )

    def update(self, next_obs, reward: float, terminated: bool, truncated: bool):
        if not isinstance(next_obs, torch.Tensor):
            next_obs = (
                torch.tensor(next_obs, dtype=torch.float32)
                .to(self.device)
                .reshape((1, -1))
            )

        # Get one-hot encoded action
        action_one_hot = self.action_oh(self.current_action)

        current_q_value = self.q_value(self.current_observation, action_one_hot)
        _, next_q_value = self.policy(next_obs)
        target = current_q_value + self.alpha * (
            reward + self.discount_factor * next_q_value - current_q_value
        )

        self.trainX = torch.cat(
            (self.trainX, torch.cat((next_obs, action_one_hot), dim=1)), dim=0
        )
        self.trainY = torch.cat(
            (
                self.trainY,
                torch.tensor(
                    [target],
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=False,
                ),
            ),
            dim=0,
        )

        # Learn at the end of every game
        if terminated:
            self.optimizer.zero_grad()

            self.trainY = self.trainY.reshape((-1, 1))
            loss = self.loss_function(self.q_network(self.trainX), self.trainY)
            loss.backward()
            self.optimizer.step()

            self.trainX = torch.tensor([], device=self.device, requires_grad=False)
            self.trainY = torch.tensor([], device=self.device, requires_grad=False)

            # Linear epsilon decay
            self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

    def evaluate_q_network(self, test_states: List[Tuple[Any, Any]]):
        if self._test_states is None:
            merged_state_action_pairs = [
                np.hstack([state.reshape((1, -1)), self.action_oh(action)])
                for state, action in test_states
            ]
            test_states = torch.tensor(
                np.array(merged_state_action_pairs), dtype=torch.float32
            )
            self._test_states = test_states

        # Average maximum Q-values for each state (makes sense to max since we use a greedy policy)
        with torch.no_grad():
            return torch.mean(torch.max(self.q_network(self._test_states)))

    def load(self, folder_path: str):
        model_path = os.path.join(folder_path, "model_weights.pth")
        self.q_network.load_state_dict(torch.load(model_path))

    def save(self, folder_path: str):
        model_path = os.path.join(folder_path, "model_weights.pth")
        torch.save(self.q_network.state_dict(), model_path)


"""
1 -1 -1 1 = 0       (1/2)
1 -1 -1 = -1        (1/3)
1 -1 1 = 1          (2/3)

number of ones + number of -1s = self.current_episode
number of ones - number of -1s = self.cummulative_reward

o + m = c
o - m = r

o = c - m
c - m - m = r

o = c - m
-2m = r - c

o = c - (c - r) / 2
m = (c - r) / 2

o = (c + r) / 2
m = (c - r) / 2
"""
