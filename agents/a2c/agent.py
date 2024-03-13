import os
import torch
import random
import logging
from torch import nn
from torch.distributions import Categorical
from copy import deepcopy
from agents.base import FootsiesAgentTorch
from gymnasium import Env
from typing import Any, Callable, Tuple
from agents.a2c.a2c import A2CLearnerBase, A2CQLearner, ActorNetwork, ValueNetwork
from agents.ql.ql import QFunctionTable, QFunctionNetwork
from agents.torch_utils import AggregateModule, observation_invert_perspective_flattened
from agents.action import ActionMap

LOGGER = logging.getLogger("main.a2c.agent")


class FootsiesAgent(FootsiesAgentTorch):
    def __init__(
        self,
        learner: A2CQLearner,
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

        self._learner = learner
        self._actor = learner.actor
        self._critic = learner.critic
        
        modules = {"actor": self._actor}
        if isinstance(self._critic, ValueNetwork):
            modules["critic"] = self._critic
        elif isinstance(self._critic, QFunctionNetwork):
            modules["critic"] = self._critic.q_network
        self._model = AggregateModule(modules)

        self.current_observation = None
        self.current_info = None
        self.current_action = None
        self.current_action_iterator = None # only needed if using simple actions

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
            self.current_action = self._learner.sample_action(obs, next_opponent_action=predicted_opponent_action)
            return self.current_action

        if predicted_opponent_action is None:
            predicted_opponent_action = random.randint(0, self.opponent_action_dim - 1)

        if self.current_action is not None:
            try:
                action = next(self.current_action_iterator)
        
            except StopIteration:
                self.current_action = None
        
        if self.current_action is None:
            self.current_action = self._learner.sample_action(obs, next_opponent_action=predicted_opponent_action)
            self.current_action_iterator = iter(ActionMap.simple_to_discrete(self.current_action))
            action = next(self.current_action_iterator)
        
        return action

    def update(self, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict):
        obs = self.current_observation

        if self.footsies:
            # We only use this method to determine whether the agent's action was frameskipped or not, and to get the opponent's action of course.
            # We treat the agent specially because we may want to use a different action space for them (e.g. remove special moves).
            obs_agent_action, obs_opponent_action = ActionMap.simples_from_torch_transition(obs, next_obs)
            obs_agent_action = obs_agent_action if obs_agent_action is None else self.current_action
            if obs_agent_action is not None and obs_agent_action != self.current_action:
                LOGGER.warning("From a transition, we determined that the agent's action was %s, but it was actually %s! There is a significant discrepancy here", obs_agent_action, self.current_action)

        else:
            obs_agent_action = self.current_action
            obs_opponent_action = None

        self._learner.learn(obs, next_obs, reward, terminated, truncated,
            obs_agent_action=obs_agent_action,
            obs_opponent_action=obs_opponent_action,
        )

        # P1 learning logging
        self.cumulative_delta += self._learner.delta
        self.cumulative_delta_n += 1
        if isinstance(self._learner, A2CQLearner):
            if self._learner.td_error is not None:
                self.cumulative_qtable_error += self._learner.td_error
                self.cumulative_qtable_error_n += 1

        # Learn using P2's perspective
        # NOTE: only valid for FOOTSIES, since both players are using the same character
        if self.footsies and self.use_opponents_perspective:
            # Switch perspectives
            p2_obs = observation_invert_perspective_flattened(obs)
            p2_next_obs = observation_invert_perspective_flattened(next_obs)

            self._learner.learn(p2_obs, p2_next_obs, reward, terminated, truncated,
                obs_agent_action=obs_agent_action,
                obs_opponent_action=obs_opponent_action,
            )

    def extract_policy(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        model = deepcopy(self._actor)

        def internal_policy(obs):
            probs = model(obs)
            return Categorical(probs=probs).sample().item()

        return super()._extract_policy(env, internal_policy)
    
    @property
    def model(self) -> nn.Module:
        return self._model
    
    @property
    def learner(self) -> A2CQLearner:
        return self._learner

    # Need to use custom save and load functions because we could use a tabular Q-function
    def load(self, folder_path: str):
        # Load actor
        actor_path = os.path.join(folder_path, "actor")
        self._actor.load_state_dict(torch.load(actor_path))
        
        # Load critic
        if isinstance(self._critic, QFunctionTable):
            critic_path = os.path.join(folder_path, "critic_qtable")
            self._critic.load(critic_path)
        elif isinstance(self._critic, QFunctionNetwork):
            critic_path = os.path.join(folder_path, "critic_qnetwork")
            self._critic.load(critic_path)
        elif isinstance(self._critic, ValueNetwork):
            critic_path = os.path.join(folder_path, "critic_vnetwork")
            self._critic.load_state_dict(torch.load(critic_path))

    def save(self, folder_path: str):
        # Save actor
        actor_path = os.path.join(folder_path, "actor")
        torch.save(self._actor.state_dict(), actor_path)
        
        # Save critic
        if isinstance(self._critic, QFunctionTable):
            critic_path = os.path.join(folder_path, "critic_qtable")
            self._critic.save(critic_path)
        elif isinstance(self._critic, QFunctionNetwork):
            critic_path = os.path.join(folder_path, "critic_qnetwork")
            self._critic.save(critic_path)
        elif isinstance(self._critic, ValueNetwork):
            critic_path = os.path.join(folder_path, "critic_vnetwork")
            torch.save(self._critic.state_dict(), critic_path)

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
            probs = self._actor(self._test_observations)
            entropies = -torch.sum(torch.log(probs + 1e-8) * probs, dim=-1)
            return torch.mean(entropies).item()
