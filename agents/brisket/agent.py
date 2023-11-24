import os

from agents.base import FootsiesAgentBase
from typing import Any, Callable, List, Tuple
import numpy as np
import torch
from torch import nn
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatten_space
from gymnasium import Env
from copy import deepcopy


class LiteralQNetwork(nn.Module):
    """This network predicts the Q-value of a state-action pair. Since the reward is either -1 or 1, the final activation layer is Tanh()"""

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        shallow: bool = True,
        shallow_size: int = 32,
        custom_final_layer: nn.Module = None,
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = (
            nn.Sequential(
                nn.Linear(n_observations + n_actions, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Tanh() if custom_final_layer is None else custom_final_layer,
            )
            if not shallow
            else nn.Sequential(
                nn.Linear(n_observations + n_actions, shallow_size),
                nn.ReLU(),
                nn.Linear(shallow_size, 1),
                nn.Tanh() if custom_final_layer is None else custom_final_layer,
            )
        )

        self._shallow = shallow
        self._has_dropout = False

    @property
    def has_dropout(self):
        return self._has_dropout

    @has_dropout.setter
    def has_dropout(self, value: bool):
        p = 0.5
        if self._has_dropout != value:
            if value:
                if self._shallow:
                    self.layers.insert(2, nn.Dropout(p=p))
                else:
                    self.layers.insert(2, nn.Dropout(p=p))
                    self.layers.insert(4, nn.Dropout(p=p))

            else:
                if self._shallow:
                    self.layers.pop(2)
                else:
                    self.layers.pop(2)
                    self.layers.pop(4)

        self._has_dropout = value

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
        shallow_size: int = 32,
        q_value_min: float = -1,
        q_value_max: float = 1,
        device: torch.device = "cpu",
        **kwargs,
    ):
        if len(kwargs) > 0:
            print(f"WARN: unknown keyword arguments for '{self.__class__.__name__}' ({kwargs})")

        self.action_space = action_space
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.device = device

        self.observations_length = flatten_space(observation_space).shape[0]
        self.actions_length = flatten_space(action_space).shape[0]
        self.q_network = LiteralQNetwork(
            self.observations_length,
            self.actions_length,
            shallow=shallow,
            shallow_size=shallow_size,
            # In FOOTSIES and FightingICE, it makes sense to have tanh as the final activation layer due to these environments' reward function
            # In other environments however, other activation functions may be more appropriate. We consider a ReLU by default for now (works for CartPole)
            custom_final_layer=nn.Tanh()
            if q_value_min == -1 and q_value_max == 1
            else nn.ReLU(),
        )

        # self.optimizer = torch.optim.Adam(
        #     self.q_network.parameters(), lr=self.learning_rate
        # )
        self.optimizer = torch.optim.SGD(
            params=self.q_network.parameters(),
            lr=self.learning_rate,
            maximize=False,
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
        self._cummulative_loss = 0
        self._cummulative_loss_n = 0

    def act(self, obs) -> Any:
        obs = self._obs_to_torch(obs)

        self.current_observation = obs
        action, _ = self.policy(obs)
        self.current_action = action
        return action

    def policy(self, obs) -> Tuple[int, float]:
        if np.random.random() < self.epsilon:
            random_action = self.action_space.sample()
            random_action_oh = self._action_onehot(random_action)
            return random_action, self.q_value(obs, random_action_oh)

        q_values = self.q_values(obs)
        return np.argmax(q_values), np.max(q_values)

    def q_values(self, obs) -> List[float]:
        return [
            self.q_value(obs, self._action_onehot(action))
            for action in range(self.actions_length)
        ]

    def q_value(self, obs, action_oh) -> float:
        with torch.no_grad():
            return self.q_network(torch.cat((obs, action_oh), dim=1)).item()

    def _action_onehot(self, action: int):
        action_onehot = np.zeros((1, self.actions_length))
        action_onehot[0, action] = 1
        return torch.tensor(
            action_onehot, dtype=torch.float32, device=self.device, requires_grad=False
        )

    def _obs_to_torch(self, obs):
        if not isinstance(obs, torch.Tensor):
            return (
                torch.tensor(obs, dtype=torch.float32).to(self.device).reshape((1, -1))
            )

        return obs

    def update(self, next_obs, reward: float, terminated: bool, truncated: bool):
        next_obs = self._obs_to_torch(next_obs)

        # Get one-hot encoded action
        action_one_hot = self._action_onehot(self.current_action)

        if terminated:
            target = reward
        else:
            current_q_value = self.q_value(self.current_observation, action_one_hot)
            _, next_q_value = self.policy(next_obs)
            target = current_q_value + self.alpha * (
                reward + self.discount_factor * next_q_value - current_q_value
            )

        self.trainX = torch.cat(
            (self.trainX, torch.cat((self.current_observation, action_one_hot), dim=1)),
            dim=0,
        )
        # NOTE: this could be computed completely at the end based on trainX
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
            output = self.q_network(self.trainX)
            loss = self.loss_function(output, self.trainY)
            loss.backward()
            self.optimizer.step()

            self.trainX = torch.tensor([], device=self.device, requires_grad=False)
            self.trainY = torch.tensor([], device=self.device, requires_grad=False)

            # Linear epsilon decay
            self.epsilon = self.epsilon - self.epsilon_decay_rate
            # Once the minimum is reached, reset back to full (avoid convergence to a local optimum)
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.epsilon_start

            # Accumulate loss as a metric
            self._cummulative_loss += loss.item()
            self._cummulative_loss_n += 1

    def _initialize_test_states(self, test_states: List[Tuple[Any, Any]]):
        if self._test_states is None:
            merged_state_action_pairs = [
                np.hstack([state.reshape((1, -1)), self._action_onehot(action)])
                for state, action in test_states
            ]
            test_states = torch.tensor(
                np.array(merged_state_action_pairs), dtype=torch.float32
            )
            self._test_states = test_states

    def evaluate_average_q_value(self, test_states: List[Tuple[Any, Any]]) -> float:
        self._initialize_test_states(test_states)

        # Average maximum Q-values for each state (makes sense to max since we use a greedy policy)
        with torch.no_grad():
            return torch.mean(torch.max(self.q_network(self._test_states)))

    def evaluate_average_action_entropy(
        self, test_states: List[Tuple[Any, Any]]
    ) -> float:
        # Average entropy of the action probability distribution obtained by softmax
        entropy = 0
        entropy_n = 0
        with torch.no_grad():
            for state, _ in test_states:
                state = self._obs_to_torch(state)
                q_values = self.q_values(state)
                exponentiated = np.exp(q_values)
                softmaxed = exponentiated / exponentiated.sum()
                entropy += -np.sum(np.log2(softmaxed) * softmaxed)
                entropy_n += 1

        return entropy / entropy_n

    # Based on: https://link.springer.com/article/10.1007/s11432-021-3347-8
    def evaluate_average_uncertainty(
        self, test_states: List[Tuple[Any, Any]], forward_passes: int = 30
    ) -> float:
        self._initialize_test_states(test_states)

        # Average uncertainty, which is the standard deviation of the Q-values after N forward passes with dropouts
        q_values = np.zeros((self._test_states.shape[0], forward_passes))
        with torch.no_grad():
            has_dropout = self.q_network.has_dropout
            self.q_network.has_dropout = True

            for i in range(forward_passes):
                q_values[:, i] = self.q_network(self._test_states).squeeze()

            self.q_network.has_dropout = has_dropout

        return np.mean(q_values.std(axis=1))

    def evaluate_average_loss_and_clear(self) -> float:
        res = (self._cummulative_loss / self._cummulative_loss_n) if self._cummulative_loss_n != 0 else 0
        self._cummulative_loss = 0
        self._cummulative_loss_n = 0
        return res

    def load(self, folder_path: str):
        model_path = os.path.join(folder_path, "model_weights.pth")
        self.q_network.load_state_dict(torch.load(model_path))

    def save(self, folder_path: str):
        model_path = os.path.join(folder_path, "model_weights.pth")
        torch.save(self.q_network.state_dict(), model_path)

    def _extract_policy(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        q_network = deepcopy(self.q_network)
        q_network.requires_grad_(False)
        
        def internal_policy(obs):
            obs = self._obs_to_torch(obs)
            
            q_values = [
                q_network(torch.cat((obs, self._action_onehot(action)), dim=1)).item()
                for action in range(self.actions_length)
            ]
            
            return np.argmax(q_values)
        
        return super()._extract_policy(env, internal_policy)