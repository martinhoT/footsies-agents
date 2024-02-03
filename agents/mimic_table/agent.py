from copy import deepcopy
import os
import numpy as np
import torch
from torch import nn
from agents.base import FootsiesAgentBase
from gymnasium import Env
from typing import Callable, List, Tuple
from gymnasium import Space
from agents.utils import FOOTSIES_ACTION_MOVES, FOOTSIES_ACTION_MOVE_INDICES_MAP


# Without environment normalization
# Guard: integers
# Move: integers
# Move progress: integers (frame)
# Position: float with tile coding


# TODO: there are three different ways we can predict actions: primitive actions discretized, primitive actions as tuple, and moves. Are all supported?
class FootsiesAgent(FootsiesAgentBase):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        by_primitive_actions: bool = False,
        use_sigmoid_output: bool = False,
        input_clip: bool = False,
        input_clip_leaky_coef: float = 0,
        hidden_layer_sizes_specification: str = "64,64",
        hidden_layer_activation_specification: str = "LeakyReLU",
        move_transition_scale: float = 10.0,
        max_allowed_loss: float = +torch.inf,
        optimize_frequency: int = 1000,
        learning_rate: float = 1e-2,
    ):
        self.by_primitive_actions = by_primitive_actions

        self.n_moves = action_space.n if by_primitive_actions else len(FOOTSIES_ACTION_MOVES)

        # Both models have the exact same structure and training regime
        player_model_kwargs = {
            "obs_size": observation_space.shape[0],
            "n_moves": self.n_moves,
            "optimize_frequency": optimize_frequency,
            "use_sigmoid_output": use_sigmoid_output,
            "input_clip": input_clip,
            "input_clip_leaky_coef": input_clip_leaky_coef,
            "hidden_layer_sizes": [int(n) for n in hidden_layer_sizes_specification.split(",")] if hidden_layer_sizes_specification else [],
            "hidden_layer_activation": getattr(nn, hidden_layer_activation_specification),
            "move_transition_scale": move_transition_scale,
            "max_allowed_loss": max_allowed_loss,
            "learning_rate": learning_rate,
        }
        self.p1_model = PlayerModel(**player_model_kwargs)
        self.p2_model = PlayerModel(**player_model_kwargs)

        self.current_observation = None

        self._test_observations = None
        self._test_actions = None
        self._p1_correct = 0
        self._p2_correct = 0
        self._random_correct = 0
        # Multiply the "correct" counters by the decay to avoid infinitely increasing counters and prioritize recent values.
        # Set to a value such that the 1000th counter value in the past will have a weight of 1%
        self._correct_decay = 0.01 ** (1 / 1000)

    def act(self, obs, p1: bool = True, deterministic: bool = False) -> "any":
        model = self.p1_model if p1 else self.p2_model
        obs = self._obs_to_tensor(obs)
        self.current_observation = obs
        return model.predict(obs, deterministic=deterministic).item()

    # TODO: change moves to include only the action moves
    def update(
        self, next_obs, reward: float, terminated: bool, truncated: bool, info: dict
    ):
        key = "action" if self.by_primitive_actions else "move"
        # This is assuming that, when using moves, all relevant moves have contiguous IDs from 0, without breaks
        # That is, the irrelevant moves (such as WIN and DEAD) have the largest IDs
        p1_move = FOOTSIES_ACTION_MOVE_INDICES_MAP[info["p1_" + key]]
        p2_move = FOOTSIES_ACTION_MOVE_INDICES_MAP[info["p2_" + key]]
        p1_move_oh = self._move_onehot(p1_move)
        p2_move_oh = self._move_onehot(p2_move)

        # TODO: evaluate only when a move transition occurs (essentially frame skipping)
        #       ... but then how do we predict when the opponent is just standing?
        self.p1_model.update(self.current_observation, p1_move_oh)
        self.p2_model.update(self.current_observation, p2_move_oh)

        # Update metrics
        self._p1_correct = (self._p1_correct * self._correct_decay) + (
            self.p1_model.predict(self.current_observation) == p1_move
        )
        self._p2_correct = (self._p2_correct * self._correct_decay) + (
            self.p2_model.predict(self.current_observation) == p2_move
        )
        self._random_correct = (self._random_correct * self._correct_decay) + (
            torch.rand((1,)).item() < 1 / self.n_moves
        )

    def _obs_to_tensor(self, obs):
        return torch.tensor(obs, dtype=torch.float32).reshape((1, -1))

    def _move_onehot(self, move: int):
        return nn.functional.one_hot(torch.tensor(move), num_classes=self.n_moves).unsqueeze(0)

    # From https://en.wikipedia.org/wiki/Hick's_law#Law
    def decision_entropy(self, obs: torch.Tensor, p1: bool) -> float:
        model = self.p1_model if p1 else self.p2_model

        probs = model.probability_distribution(obs)

        logs = torch.log2(1 / probs + 1)
        logs = torch.nan_to_num(logs, 0)  # in case there were 0 probabilities
        return torch.sum(probs * logs)

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

    def evaluate_performance(self, p1: bool) -> float:
        """Evaluate relative performance against a random predictor (how many times does it predict better than a random predictor)"""
        correct = self._p1_correct if p1 else self._p2_correct
        return correct / self._random_correct

    def _initialize_test_states(self, test_states: List[torch.Tensor]):
        if self._test_observations is None or self._test_actions is None:
            # If the state is terminal, then there will be no action. Discard those states
            observations, actions = map(
                np.array, zip(*filter(lambda sa: sa[1] is not None, test_states))
            )
            self._test_observations = torch.tensor(observations, dtype=torch.float32)
            self._test_actions = torch.tensor(actions, dtype=torch.float32)

    def evaluate_divergence_between_players(
        self, test_states: List[torch.Tensor]
    ) -> float:
        self._initialize_test_states(test_states)

        p1_predicted = self.p1_model.predict(
            self._test_observations, deterministic=True
        )
        p2_predicted = self.p2_model.predict(
            self._test_observations, deterministic=True
        )

        return torch.sum(p1_predicted == p2_predicted) / self._test_actions.size(0)

    def evaluate_decision_entropy(
        self, test_states: List[torch.Tensor], p1: bool
    ) -> float:
        self._initialize_test_states(test_states)

        return self.decision_entropy(self._test_observations, p1=p1)

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
