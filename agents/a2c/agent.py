import os
import numpy as np
import torch
import random
from torch import nn
from torch.distributions import Categorical
from copy import deepcopy
from agents.base import FootsiesAgentTorch
from gymnasium import Env
from typing import Any, Callable, Tuple
from agents.a2c.a2c import A2CLearnerBase, A2CQLearner
from agents.ql.ql import QFunctionTable
from agents.torch_utils import AggregateModule, observation_invert_perspective_flattened
from agents.action import ActionMap


class FootsiesAgent(FootsiesAgentTorch):
    def __init__(
        self,
        learner: A2CLearnerBase,
        opponent_action_dim: int,
        footsies: bool = True,
        use_opponents_perspective: bool = False,
    ):
        """
        Footsies agent using the A2C algorithm, potentially with some modifications.
        This implementation assumes simplified actions.

        Parameters
        ----------
        - `opponent_action_dim`: the number of opponent actions
        - `learner`: the A2C algorithm class to use. If `None`, one will be created
        - `footsies`: whether to consider the FOOTSIES environment is being used. If `False`, the agent will not do any special treatment
        - `use_opponents_perspective`: whether to use the opponent's perspective for learning.
        Only valid for FOOTSIES, since it's an environment with 2 players on the same conditions (characters, more specifically)
        """
        # NOTE: we *could* use the opponent's perspective if we use the opponent model as the policy, but that makes this whole thing more iffy than it already is
        if use_opponents_perspective:
            raise NotImplementedError("using the opponent's perspective for learning is not supported, mainly because it has not been figured out if it's valid or not")

        self.opponent_action_dim = opponent_action_dim
        self.footsies = footsies
        self.use_opponents_perspective = use_opponents_perspective

        self.learner = learner
        self.actor = learner._actor
        self.critic = learner._critic

        models = {"actor": self.actor}
        # Could be a QTable
        if isinstance(self.critic, nn.Module):
            models["critic"] = self.critic
        self.actor_critic = AggregateModule(models)

        self.current_observation = None
        self.current_info = None
        self.current_action = None

        # For logging
        self.cumulative_delta = 0
        self.cumulative_delta_n = 0
        self.cumulative_qtable_error = 0
        self.cumulative_qtable_error_n = 0
        self._test_observations = None

    def act(self, obs: torch.Tensor, info: dict, predicted_opponent_action: int = None) -> "any":
        self.current_observation = obs
        self.current_info = info
        
        # Perform the normal action selection not considering FOOTSIES and be done with it
        if not self.footsies:
            self.current_action = self.learner.sample_action(obs, next_opponent_action=predicted_opponent_action)
            return self.current_action

        if predicted_opponent_action is None:
            predicted_opponent_action = random.randint(0, self.opponent_action_dim - 1)

        if self.current_action is not None:
            try:
                action = next(self.current_action)
        
            except StopIteration:
                self.current_action = None
        
        if self.current_action is None:
            simple_action = self.learner.sample_action(obs, next_opponent_action=predicted_opponent_action)
            self.current_action = iter(ActionMap.simple_to_discrete(simple_action))
            action = next(self.current_action)
        
        return action

    def update(self, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict):
        obs = self.current_observation

        # NOTE: with the way frameskipping is being done we are updating Q-values throughout the duration of moves, not "skipping" them
        if self.footsies:
            p1_move_progress = obs[0, 32].item()
            p2_move_progress = obs[0, 33].item()
            p1_move_index = torch.argmax(obs[0, 2:17]).item()
            p2_move_index = torch.argmax(obs[0, 17:32]).item()
            p1_next_move_index = torch.argmax(next_obs[0, 2:17]).item()
            p2_next_move_index = torch.argmax(next_obs[0, 17:32]).item()
            obs_agent_action = ActionMap.simple_from_transition(
                previous_player_move_index=p1_move_index,
                previous_opponent_move_index=p2_move_index,
                previous_player_move_progress=p1_move_progress,
                previous_opponent_move_progress=p2_move_progress,
                player_move_index=p1_next_move_index,
            )
            obs_opponent_action = ActionMap.simple_from_transition(
                previous_player_move_index=p2_move_index,
                previous_opponent_move_index=p1_move_index,
                previous_player_move_progress=p2_move_progress,
                previous_opponent_move_progress=p1_move_progress,
                player_move_index=p2_next_move_index,
            )

        else:
            obs_agent_action = self.current_action
            obs_opponent_action = None

        self.learner.learn(obs, next_obs, reward, terminated, truncated,
            obs_agent_action=obs_agent_action,
            obs_opponent_action=obs_opponent_action,
        )

        # P1 learning logging
        self.cumulative_delta += self.learner.delta
        self.cumulative_delta_n += 1
        if isinstance(self.learner, A2CQLearner):
            if self.learner.td_error is not None:
                self.cumulative_qtable_error += self.learner.td_error
                self.cumulative_qtable_error_n += 1

        # Learn using P2's perspective
        # NOTE: only valid for FOOTSIES, since both players are using the same character
        if self.footsies and self.use_opponents_perspective:
            # Switch perspectives
            p2_obs = observation_invert_perspective_flattened(obs)
            p2_next_obs = observation_invert_perspective_flattened(next_obs)

            self.learner.learn(p2_obs, p2_next_obs, reward, terminated, truncated,
                obs_agent_action=obs_agent_action,
                obs_opponent_action=obs_opponent_action,
            )

    def extract_policy(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        model = deepcopy(self.actor_critic.actor)

        def internal_policy(obs):
            probs = model(obs)
            return Categorical(probs=probs).sample().item()

        return super()._extract_policy(env, internal_policy)
    
    @property
    def model(self) -> nn.Module:
        return self.actor_critic
    
    # Need to use custom save and load functions because we could use a tabular Q-function
    def load(self, folder_path: str):
        model_path = os.path.join(folder_path, "model")
        self.model.load_state_dict(torch.load(model_path))
        if isinstance(self.critic, QFunctionTable):
            qtable_path = os.path.join(folder_path, "q")
            self.critic.load(qtable_path)

    def save(self, folder_path: str):
        model_path = os.path.join(folder_path, "model")
        torch.save(self.model.state_dict(), model_path)
        if isinstance(self.critic, QFunctionTable):
            qtable_path = os.path.join(folder_path, "q")
            self.critic.save(qtable_path)

    def evaluate_average_delta(self) -> float:
        res = (
            self.cumulative_delta / self.cumulative_delta_n
        ) if self.cumulative_delta_n != 0 else 0

        self.cumulative_delta = 0
        self.cumulative_delta_n = 0

        return res
        
    def evaluate_average_qtable_error(self) -> float:
        res = (
            self.cumulative_qtable_error / self.cumulative_qtable_error_n
        ) if self.cumulative_qtable_error_n != 0 else 0

        self.cumulative_qtable_error = 0
        self.cumulative_qtable_error_n = 0

        return res

    def _initialize_test_states(self, test_states: list[torch.Tensor]):
        if self._test_observations is None:
            test_observations, _ = zip(*test_states)
            self._test_observations = torch.vstack(test_observations)

    def evaluate_average_policy_entropy(self, test_states: list[tuple[Any, Any]]) -> float:
        self._initialize_test_states(test_states)

        with torch.no_grad():
            probs = self.actor(self._test_observations)
            entropies = -torch.sum(torch.log(probs + 1e-8) * probs, dim=-1)
            return torch.mean(entropies).item()
