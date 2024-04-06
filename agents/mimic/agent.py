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
from agents.logger import TestState
from agents.mimic.mimic import PlayerModel

LOGGER = logging.getLogger("main.mimic.agent")


# TODO: there are three different ways we can predict actions: primitive actions discretized, primitive actions as tuple, and moves. Are all supported?
class MimicAgent(FootsiesAgentBase):
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

        self._test_observations = None
        self._test_p1_actions = None
        self._test_p2_actions = None
        self._test_observations_partitioned = None
        self._test_p1_actions_partitioned = None
        self._test_p2_actions_partitioned = None

        # Logging
        self._p1_cumulative_loss = 0
        self._p2_cumulative_loss = 0
        self._p1_cumulative_loss_n = 0
        self._p2_cumulative_loss_n = 0

    def act(self, obs: torch.Tensor, info: dict, p1: bool = True, deterministic: bool = False, predict: bool = False) -> "any":
        if predict:
            model = self._p1_model if p1 else self._p2_model
            prediction = model.predict(obs, deterministic=deterministic).item()
            return prediction

        return 0

    def update_with_simple_actions(self, obs: torch.Tensor, p1_simple: int | None, p2_simple: int | None, terminated_or_truncated: bool):
        """Perform an update with the given simple actions, useful to avoid recomputing them."""
        if self._learn_p1:
            loss = self._p1_model.update(obs, p1_simple, terminated_or_truncated, 1.0)
            if loss is not None:
                self._p1_cumulative_loss += loss
                self._p1_cumulative_loss_n += 1
        
        if self._learn_p2:
            loss = self._p2_model.update(obs, p2_simple, terminated_or_truncated, 1.0)
            if loss is not None:
                self._p2_cumulative_loss += loss
                self._p2_cumulative_loss_n += 1

    def update(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict):
        p1_simple, p2_simple = ActionMap.simples_from_transition_ori(info, next_info)

        self.update_with_simple_actions(obs, p1_simple, p2_simple, terminated or truncated)

    def decision_entropy(self, obs: torch.Tensor, p1: bool) -> torch.Tensor:
        """The decision entropy of the player model at the given observation."""
        model = self._p1_model if p1 else self._p2_model
        dist = model.network.distribution(obs)
        return dist.entropy()

    def load(self, folder_path: str):
        if self.p1_model is not None:
            p1_path = os.path.join(folder_path, "p1")
            self._p1_model.load(p1_path)
        if self.p2_model is not None:
            p2_path = os.path.join(folder_path, "p2")
            self._p2_model.load(p2_path)

    def save(self, folder_path: str):
        if self.p1_model is not None:
            p1_path = os.path.join(folder_path, "p1")
            self._p1_model.save(p1_path)
        if self.p2_model is not None:
            p2_path = os.path.join(folder_path, "p2")
            self._p2_model.save(p2_path)

    def evaluate_p1_average_loss_and_clear(self) -> float:
        res = (
            (self._p1_cumulative_loss / self._p1_cumulative_loss_n)
            if self._p1_cumulative_loss_n != 0
            else None
        )

        self._p1_cumulative_loss = 0
        self._p1_cumulative_loss_n = 0
        return res

    def evaluate_p2_average_loss_and_clear(self) -> float:
        res = (
            (self._p2_cumulative_loss / self._p2_cumulative_loss_n)
            if self._p2_cumulative_loss_n != 0
            else None
        )

        self._p2_cumulative_loss = 0
        self._p2_cumulative_loss_n = 0
        return res

    def _initialize_test_states(self, test_states: List[TestState]):
        if self._test_observations is None:
            # Only consider observations in which an action was performed (this is not the case, for instance, when the environment terminates)
            test_observations, test_p1_actions, test_p2_actions = zip((s.observation, s.p1_action_simple, s.p2_action_simple) for s in test_states if not (s.terminated or s.truncated))
            self._test_observations = torch.vstack(test_observations)
            self._test_p1_actions = torch.tensor(test_p1_actions, dtype=torch.long).view(-1, 1)
            self._test_p2_actions = torch.tensor(test_p2_actions, dtype=torch.long).view(-1, 1)
            
            # We assume the test state were obtained sequentially.
            # NOTE: episode length doesn't count the terminal state, which is not kept.
            episode_lengths = []
            length = 0
            for test_state in test_states:
                if (test_state.terminated or test_state.truncated) and length > 0:
                    episode_lengths.append(length)
                    length = 0
    
                else:
                    length += 1
            
            # In case the last episode didn't complete fully, and so wasn't added inside the loop
            if length > 0:
                episode_lengths.append(length)
            
            self._test_observations_partitioned = torch.split(self._test_observations, episode_lengths)
            self._test_p1_actions_partitioned = torch.split(self._test_p1_actions, episode_lengths)
            self._test_p2_actions_partitioned = torch.split(self._test_p2_actions, episode_lengths)

    def evaluate_divergence_between_players(self, test_states: List[TestState]) -> float:
        """
        Kullback-Leibler divergence between player 1 and player 2 (i.e. the excess surprise of using player 2's model when it is actually player 1's).
        
        If both players are the same, it's expected that this divergence is 0.
        """
        self._initialize_test_states(test_states)

        with torch.no_grad():
            p1_probs = self.p1_model.network.log_probabilities(self._test_observations)[0].detach()
            p2_probs = self.p2_model.network.log_probabilities(self._test_observations)[0].detach()
            divergence = nn.functional.kl_div(p2_probs, p1_probs, reduction="batchmean", log_target=True)

        return divergence

    def evaluate_decision_entropy(self, test_states: List[TestState], p1: bool) -> float:
        """Evaluate the entropy of the predicted probability distribution at the given test states."""
        self._initialize_test_states(test_states)

        return self.decision_entropy(self._test_observations, p1=p1).mean().item()

    def evaluate_prediction_score(self, test_states: List[TestState], p1: bool):
        """
        Evaluate the prediction score of the player model at the given test states.
        
        The prediction score is the sum of probabilities of the true actions under the predicted distribution, over the total number of test states.
        Should be as close to 1 as possible.
        """
        self._initialize_test_states(test_states)

        model = self.p1_model if p1 else self.p2_model

        total_score = 0
        for (observations, p1_actions, p2_actions) in zip(self._test_observations_partitioned, self._test_p1_actions_partitioned, self._test_p2_actions_partitioned):
            probs, _ = model.network.probabilities(observations, None)
            p1_score = probs.gather(1, p1_actions).sum().item()
            p2_score = probs.gather(1, p2_actions).sum().item()

            total_score += p1_score + p2_score

        # We are making 2 predictions at each observation, so the denominator is multiplied by 2.
        return total_score / (2 * len(test_states))

    # NOTE: this only works if the class was defined to be over primitive actions
    def extract_policy(self, env: Env, use_p1: bool = True) -> Callable[[dict], Tuple[bool, bool, bool]]:
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
