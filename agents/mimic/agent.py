import os
import numpy as np
import torch
import logging
from copy import deepcopy
from torch import nn
from agents.base import FootsiesAgentBase
from agents.action import ActionMap
from gymnasium import Env
from typing import Any, Callable, List, Tuple
from agents.mimic.mimic import PlayerModel

LOGGER = logging.getLogger("main.mimic.agent")


# TODO: there are three different ways we can predict actions: primitive actions discretized, primitive actions as tuple, and moves. Are all supported?
class FootsiesAgent(FootsiesAgentBase):
    def __init__(
        self,
        action_dim: int,
        by_primitive_actions: bool = False,
        p1_model: PlayerModel = None,
        p2_model: PlayerModel = None,
    ):
        """
        FOOTSIES agent that builds a player model of either one of or both players.
        At least one player model should be specified.
        It doesn't make sense otherwise.

        NOTE: the player models will be learnt using the same observation for simplicity, and to avoid
        constant observation manipulation. If, for instance, player 2's model is to be used as player 1's,
        then the observation's perspective should be shifted first before interacting with that model.

        Parameters
        ----------
        - `action_dim`: the dimensionality of the action space. Will be ignored if not by primitive actions
        - `by_primitive_actions`: whether to predict primitive actions rather than simple ones
        - `p1_model`: player 1's model. If `None`, then no model is learnt for tihs player
        - `p2_model`: player 2's model. If `None`, then no model is learnt for this player
        """
        if by_primitive_actions:
            raise NotImplementedError("imitating primitive actions is not supported yet")
        
        if p1_model is None and p2_model is None:
            raise ValueError("at least one model should be learnt, but both players' models are None")

        self.by_primitive_actions = by_primitive_actions
        self._p1_model = p1_model
        self._p2_model = p2_model
        self._learn_p1 = p1_model is not None
        self._learn_p2 = p2_model is not None

        self.n_moves = action_dim if by_primitive_actions else ActionMap.n_simple()

        self.current_observation = None
        self.current_info = None

        self._test_observations = None
        self._test_actions = None

        # Logging
        self._p1_cumulative_loss = 0
        self._p2_cumulative_loss = 0
        self._p1_cumulative_loss_n = 0
        self._p2_cumulative_loss_n = 0

    def act(self, obs: torch.Tensor, info: dict, p1: bool = True, deterministic: bool = False, predict: bool = False) -> "any":
        self.current_observation = obs
        self.current_info = info

        if predict:
            model = self._p1_model if p1 else self._p2_model
            prediction = model.predict(obs, deterministic=deterministic).item()
            return prediction

        return 0

    def update(self, next_obs, reward: float, terminated: bool, truncated: bool, info: dict):
        p1_simple, p2_simple = ActionMap.simples_from_transition_ori(self.current_info, info)

        if self._learn_p1 and p1_simple is not None:
            loss = self._p1_model.update(self.current_observation, p1_simple, 1.0)
            self._p1_cumulative_loss += loss
            self._p1_cumulative_loss_n += 1
        
        if self._learn_p2 and p2_simple is not None:
            loss = self._p2_model.update(self.current_observation, p2_simple, 1.0)
            self._p2_cumulative_loss += loss
            self._p2_cumulative_loss += 1

    def decision_entropy(self, obs: torch.Tensor, p1: bool) -> float:
        """The decision entropy of the player model at the given observation."""
        model = self._p1_model if p1 else self._p2_model
        dist = model.network.distribution(obs)
        return dist.entropy()

    def load(self, folder_path: str):
        p1_path = os.path.join(folder_path, "p1")
        p2_path = os.path.join(folder_path, "p2")
        self._p1_model.load(p1_path)
        self._p2_model.load(p2_path)

    def save(self, folder_path: str):
        p1_path = os.path.join(folder_path, "p1")
        p2_path = os.path.join(folder_path, "p2")
        self._p1_model.save(p1_path)
        self._p2_model.save(p2_path)

    def evaluate_p1_average_loss_and_clear(self, p1: bool) -> float:
        res = (
            (self._p1_cumulative_loss / self._p1_cumulative_loss_n)
            if self._p1_cumulative_loss_n != 0
            else 0
        )

        self._p1_cumulative_loss = 0
        self._p1_cumulative_loss_n = 0
        return res

    def evaluate_p2_average_loss_and_clear(self) -> float:
        res = (
            (self._p2_cumulative_loss / self._p2_cumulative_loss_n)
            if self._p2_cumulative_loss_n != 0
            else 0
        )

        self._p2_cumulative_loss = 0
        self._p2_cumulative_loss_n = 0
        return res

    def _initialize_test_states(self, test_states: List[torch.Tensor]):
        if self._test_observations is None or self._test_actions is None:
            # If the state is terminal, then there will be no action. Discard those states
            observations, actions = map(
                np.array, zip(*filter(lambda sa: sa[1] is not None, test_states))
            )
            self._test_observations = torch.tensor(observations, dtype=torch.float32)
            self._test_actions = torch.tensor(actions, dtype=torch.float32)

    def evaluate_divergence_between_players(
        self, test_states: List[tuple[Any, Any]]
    ) -> float:
        """
        Kullback-Leibler divergence between player 1 and player 2 (i.e. the excess surprise of using player 2's model when it is actually player 1's).
        
        If both players are the same, it's expected that this divergence is 0.
        """
        self._initialize_test_states(test_states)

        p1_probs = self.p1_model.network.probabilities(self._test_observations)
        p2_probs = self.p2_model.network.probabilities(self._test_observations)
        divergence = nn.functional.kl_div(p1_probs, p2_probs)

        return divergence

    def evaluate_decision_entropy(
        self, test_states: List[torch.Tensor], p1: bool
    ) -> float:
        self._initialize_test_states(test_states)

        return self.decision_entropy(self._test_observations, p1=p1)

    # NOTE: this only works if the class was defined to be over primitive actions
    def extract_policy(
        self, env: Env, use_p1: bool = True
    ) -> Callable[[dict], Tuple[bool, bool, bool]]:
        model_to_use = self._p1_model if use_p1 else self._p2_model
        obs_mask = self._p1_model.obs_mask if use_p1 else self._p2_model.obs_mask

        policy_network = deepcopy(model_to_use._network)
        policy_network.requires_grad_(False)

        def internal_policy(obs):
            obs = self._obs_to_torch(obs)
            predicted = policy_network(obs[:, obs_mask])
            return torch.argmax(predicted)

        return super()._extract_policy(env, internal_policy)

    @property
    def p1_model(self) -> PlayerModel:
        """Player 1's model."""
        return self._p1_model

    @property
    def p2_model(self) -> PlayerModel:
        """Player 2's model."""
        return self._p2_model

    @property
    def learn_p1(self) -> bool:
        """Whether a model of player 1 is being learnt. Can be toggled."""
        return self._learn_p1

    @property
    def learn_p2(self) -> bool:
        """Whether a model of player 2 is being learnt. Can be toggled."""
        return self._learn_p2
    
    @learn_p1.setter
    def learn_p1(self, value: bool):
        self._learn_p1 = value
    
    @learn_p2.setter
    def learn_p2(self, value: bool):
        self._learn_p2 = value
