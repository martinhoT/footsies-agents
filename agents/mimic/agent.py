from copy import deepcopy
import os
import numpy as np
import torch
from torch import nn
from agents.base import FootsiesAgentBase
from gymnasium import Env
from typing import Any, Callable, List, Tuple
from gymnasium import Space
from footsies_gym.moves import FootsiesMove


RELEVANT_MOVES = set(FootsiesMove) - {FootsiesMove.WIN, FootsiesMove.DEAD}


# Sigmoid output is able to tackle action combinations
class PlayerModelNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, output_sigmoid: bool = False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            # nn.Linear(32, 32),
            nn.Linear(32, output_dim),
            nn.Sigmoid() if output_sigmoid else nn.Softmax(0),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class PlayerModel:
    def __init__(self, obs_size: int, n_moves: int, optimize_frequency: int = 1000):
        self.optimize_frequency = optimize_frequency

        self.network = PlayerModelNetwork(obs_size, n_moves)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(params=self.network.parameters(), lr=1e2)

        self.x_batch_as_list = []
        self.y_batch_as_list = []
        self.step = 0
        self.cummulative_loss = 0
        self.cummulative_loss_n = 0

    def update(self, obs: torch.Tensor, action: torch.Tensor):
        self.x_batch_as_list.append(obs)
        self.y_batch_as_list.append(action)
        self.step += 1

        if self.step >= self.optimize_frequency:
            x_batch = torch.cat(self.x_batch_as_list)
            y_batch = torch.stack(self.y_batch_as_list)
            error = None

            # Keep training on examples that are problematic
            while error is None or error > 0:
                self.optimizer.zero_grad()

                predicted = self.network(x_batch)
                loss = self.loss_function(predicted, y_batch)

                loss.backward()
                self.optimizer.step()

                # TODO: don't try this for now
                # error = (loss > ...).sum()
                error = 0

                self.cummulative_loss += loss.item()
                self.cummulative_loss_n += 1

            self.x_batch_as_list.clear()
            self.y_batch_as_list.clear()
            self.step = 0

    def predict(self, obs: torch.Tensor) -> "any":
        with torch.no_grad():
            return torch.argmax(self.network(obs), axis=1)

    def load(self, path: str):
        self.network.load_state_dict(torch.load(path))

    def save(self, path: str):
        torch.save(self.network.state_dict(), path)


# TODO: there are three different ways we can predict actions: primitive actions discretized, primitive actions as tuple, and moves. Are all supported?
class FootsiesAgent(FootsiesAgentBase):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        over_primitive_actions: bool = False,
        optimize_frequency: int = 1000,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.over_primitive_actions = over_primitive_actions

        self.n_moves = action_space.n if over_primitive_actions else len(RELEVANT_MOVES)

        self.p1_model = PlayerModel(
            observation_space.shape[0], self.n_moves, optimize_frequency
        )
        self.p2_model = PlayerModel(
            observation_space.shape[0], self.n_moves, optimize_frequency
        )

        self.current_observation = None

        self._test_observations = None
        self._test_actions = None

    def act(self, obs) -> "any":
        obs = self._obs_to_tensor(obs)
        self.current_observation = obs
        return self.p1_model.predict(obs)

    def update(
        self, next_obs, reward: float, terminated: bool, truncated: bool, info: dict
    ):
        key = "action" if self.over_primitive_actions else "move"
        # This is assuming that, when using moves, all relevant moves have contiguous IDs from 0, without breaks
        # That is, the irrelevant moves (WIN and DEAD) have the largest IDs
        p1_move = self._move_onehot(info["p1_" + key])
        p2_move = self._move_onehot(info["p2_" + key])

        self.p1_model.update(self.current_observation, p1_move)
        self.p2_model.update(self.current_observation, p2_move)

    def _obs_to_tensor(self, obs):
        return torch.tensor(obs, dtype=torch.float32).reshape((1, -1))

    def _move_onehot(self, move: int):
        onehot = torch.zeros((self.n_moves,))
        onehot[move] = 1
        return onehot

    def load(self, folder_path: str):
        p1_path = os.path.join(folder_path, "p1")
        p2_path = os.path.join(folder_path, "p2")
        self.p1_model.save(p1_path)
        self.p2_model.save(p2_path)

    def save(self, folder_path: str):
        p1_path = os.path.join(folder_path, "p1")
        p2_path = os.path.join(folder_path, "p2")
        self.p1_model.save(p1_path)
        self.p2_model.save(p2_path)

    def evaluate_average_loss_and_clear(self, p1: bool) -> float:
        model = self.p1_model if p1 else self.p2_model

        res = (
            (model.cummulative_loss / model.cummulative_loss_n)
            if model.cummulative_loss_n != 0
            else 0
        )

        model.cummulative_loss = 0
        model.cummulative_loss_n = 0
        return res

    def _initialize_test_states(self, test_states):
        if self._test_observations is None or self._test_actions is None:
            # If the state is terminal, then there will be no action. Discard those states
            observations, actions = map(np.array,
                zip(*filter(lambda sa: sa[1] is not None, test_states))
            )
            self._test_observations = torch.tensor(observations, dtype=torch.float32)
            self._test_actions = torch.tensor(actions, dtype=torch.float32)

    def evaluate_divergence_between_players(self, test_states) -> float:
        self._initialize_test_states(test_states)

        p1_predicted = self.p1_model.predict(self._test_observations)
        p2_predicted = self.p2_model.predict(self._test_observations)

        return torch.sum(p1_predicted == p2_predicted) / self._test_actions.size(0)

    def evaluate_accuracy(self, test_states: List[Tuple[Any, Any]]) -> float:
        self._initialize_test_states(test_states)

        predicted = self.p1_model.predict(self._test_observations)

        return torch.sum(predicted == self._test_actions) / self._test_actions.size(0)

    # NOTE: this only works if the class was defined to be over primitive actions
    def extract_policy(
        self, env: Env, use_p1: bool = True
    ) -> Callable[[dict], Tuple[bool, bool, bool]]:
        model_to_use = self.p1_model if use_p1 else self.p2_model

        policy_network = deepcopy(model_to_use.network)
        policy_network.requires_grad_(False)

        def internal_policy(obs):
            obs = self._obs_to_torch(obs)
            predicted = policy_network(obs)
            return torch.argmax(predicted)

        return super()._extract_policy(env, internal_policy)
