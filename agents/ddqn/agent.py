from agents.base import FootsiesAgentBase
from typing import Any, Iterator
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatten_space
from collections import namedtuple, deque


class QNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 16), nn.Tanh(), nn.Linear(16, output_size)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits


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
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        replay_memory_capacity: int = 10000,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.9,
        epsilon_decay_rate: float = 0.001,
        min_epsilon: float = 0.05,
        update_frequency: int = 300,
        target_update_rate: float = 0.01,
        log_run: bool = True,
        device: torch.device = "cpu",
        **kwargs,
    ):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.update_frequency = update_frequency
        self.target_update_rate = target_update_rate
        self.log_run = log_run
        self.device = device

        flattened_observation_size = flatten_space(observation_space).shape[0]
        flattened_action_size = flatten_space(action_space).shape[0]
        self.policy_network = QNetwork(
            flattened_observation_size, flattened_action_size
        ).to(device)
        self.target_network = QNetwork(
            flattened_observation_size, flattened_action_size
        ).to(device)
        self.replay_memory = ReplayMemory(replay_memory_capacity)

        self.optimizer = torch.optim.SGD(
            self.policy_network.parameters(), lr=learning_rate
        )
        # self.optimizer = torch.optim.RMSprop(
        #     self.policy_network.parameters(), lr=learning_rate
        # )
        self.loss_function = lambda output, target, reward: (
            reward + discount_factor * target - output
        ).clip(
            -1, 1
        )  # clipped

        self.current_iteration = 0
        self.current_observation = None
        self.current_action = None

        self.summary_writer = SummaryWriter() if self.log_run else None
        self.cummulative_reward = 0
        self.current_step = 0

    def act(self, obs) -> Any:
        if not isinstance(obs, torch.Tensor):
            obs = (
                torch.tensor(obs, dtype=torch.float32).to(self.device).reshape((1, -1))
            )

        self.current_observation = obs
        action = self.policy(obs)
        self.current_action = action
        return action

    def policy(self, obs) -> Any:
        # Exploration
        if np.random.random() < self.epsilon:
            return self.action_space.sample()

        # Greedy
        with torch.no_grad():  # don't calculate gradients
            logits = self.policy_network(obs)
        # return nn.Tanh()(logits) > 0.0
        return logits.argmax(dim=1).item()

    def update(self, next_obs, reward: float, terminated: bool, truncated: bool):
        if not isinstance(next_obs, torch.Tensor):
            next_obs = (
                torch.tensor(next_obs, dtype=torch.float32)
                .to(self.device)
                .reshape((1, -1))
            )

        transition = Transition(
            self.current_observation, self.current_action, next_obs, reward
        )
        self.replay_memory.add_transition(transition)

        # Update networks
        if self.current_iteration == self.update_frequency - 1:
            self.optimizer.zero_grad()  # prevent double-counting

            # TODO: instead of `for` cycle use vectorized operations
            for (
                t_obs,
                t_action,
                t_next_obs,
                t_reward,
            ) in self.replay_memory.random_sample():
                output_logits = self.policy_network(t_obs)
                target_logits = self.target_network(t_next_obs)

                # Because actions are combinations of the final network output, we need to this a bit differently.
                # Due to this, the greedy action selection is defined differently
                # output = output_logits[t_action].sum()
                # target = target_logits[
                #     target_logits > 0.0
                # ].sum()  # since it's a combination of actions
                output = output_logits[:, t_action]
                target, _ = target_logits.max(dim=1)

                loss = self.loss_function(output, target, t_reward)
                loss.backward()

                # Kills performance
                # self.summary_writer.add_graph(self.policy_network, t_obs)
                # self.summary_writer.add_graph(self.target_network, t_next_obs)

            self.optimizer.step()

            new_target_state_dict = self.target_network.state_dict()
            for k, v_policy in self.policy_network.state_dict().items():
                new_target_state_dict[
                    k
                ] = v_policy * self.target_update_rate + new_target_state_dict[k] * (
                    1 - self.target_update_rate
                )
            self.target_network.load_state_dict(new_target_state_dict)

            self.epsilon = self.epsilon + self.epsilon_decay_rate * (
                self.min_epsilon - self.epsilon
            )

        self.current_iteration = (self.current_iteration + 1) % self.update_frequency

        self.cummulative_reward += reward
        self.current_step += 1
        if self.log_run:
            self.summary_writer.add_scalar(
                "Reward", self.cummulative_reward, self.current_step
            )
            self.summary_writer.add_scalar(
                "Win rate",
                (self.current_step + self.cummulative_reward) / (2 * self.current_step),
            )
