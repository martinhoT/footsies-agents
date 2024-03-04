import os
import torch
import random
from torch import nn
from torch.distributions import Categorical
from copy import deepcopy
from agents.base import FootsiesAgentTorch
from gymnasium import Env
from typing import Any, Callable, Tuple
from agents.a2c.a2c import A2CLambdaLearner, ActorNetwork, CriticNetwork, A2CLearnerBase, A2CQLearner
from agents.ql.ql import QTable
from agents.utils import extract_sub_kwargs
from agents.torch_utils import AggregateModule
from agents.action import ActionMap
from footsies_gym.moves import FOOTSIES_MOVE_INDEX_TO_MOVE


class FootsiesAgent(FootsiesAgentTorch):
    def __init__(
        self,
        observation_space_size: int,
        action_space_size: int,
        learner: A2CLearnerBase = None,
        use_simple_actions: bool = True,
        use_q_table: bool = False,
        actor_hidden_layer_sizes_specification: str = "",
        critic_hidden_layer_sizes_specification: str = "",
        actor_hidden_layer_activation_specification: str = "ReLU",
        critic_hidden_layer_activation_specification: str = "ReLU",
        **kwargs,
    ):
        """
        Footsies agent using the A2C algorithm, potentially with some modifications.

        Parameters
        ----------
        - `observation_space_size`: the size of the observation space
        - `action_space_size`: the size of the action space
        - `learner`: the A2C algorithm class to use. If `None`, one will be created
        - `use_simple_actions`: whether to use simple actions rather than discrete actions
        - `use_q_table`: whether to use a Q-table instead of a neural network for the critic
        - `actor_hidden_layer_sizes_specification`: a string specifying the hidden layer sizes for the actor network
        - `critic_hidden_layer_sizes_specification`: a string specifying the hidden layer sizes for the critic network
        - `actor_hidden_layer_activation_specification`: a string specifying the hidden layer activation for the actor network
        - `critic_hidden_layer_activation_specification`: a string specifying the hidden layer activation for the critic network
        """
        self.action_space_size = action_space_size if use_simple_actions else ActionMap.n_simple()
        self.use_simple_actions = use_simple_actions

        a2c_kwargs, actor_kwargs, critic_kwargs = extract_sub_kwargs(kwargs, ("a2c", "actor", "critic"), strict=True)

        if learner is None:
            actor = ActorNetwork(
                obs_dim=observation_space_size,
                action_dim=self.action_space_size,
                hidden_layer_sizes=[int(n) for n in actor_hidden_layer_sizes_specification.split(",")] if actor_hidden_layer_sizes_specification else [],
                hidden_layer_activation=getattr(nn, actor_hidden_layer_activation_specification),
                **actor_kwargs,
            )

            if use_q_table:
                critic = QTable(
                    action_dim=self.action_space_size,
                    opponent_action_dim=self.action_space_size,
                    **critic_kwargs,
                )
                learner_class = A2CQLearner
            else:
                critic = CriticNetwork(
                    obs_dim=observation_space_size,
                    hidden_layer_sizes=[int(n) for n in critic_hidden_layer_sizes_specification.split(",")] if critic_hidden_layer_sizes_specification else [],
                    hidden_layer_activation=getattr(nn, critic_hidden_layer_activation_specification),
                    **critic_kwargs,
                )
                learner_class = A2CLambdaLearner

            learner = learner_class(
                actor=actor,
                critic=critic,
                **a2c_kwargs,
            )

        self.learner = learner
        self.actor = learner.actor
        self.critic = learner.critic

        models = {"actor": self.actor}
        # Could be a QTable
        if isinstance(self.critic, nn.Module):
            models["critic"] = self.critic
        self.actor_critic = AggregateModule(models)

        self.current_observation = None
        self.current_info = None

        # For logging
        self.cumulative_delta = 0
        self.cumulative_delta_n = 0
        self._test_observations = None

    def act(self, obs: torch.Tensor, info: dict, predicted_opponent_action: int = None) -> "any":
        self.current_observation = obs
        self.current_info = info
        
        if predicted_opponent_action is None:
            predicted_opponent_action = random.randint(0, self.action_space_size - 1)

        if self.current_action is not None:
            try:
                action = next(self.current_action)
        
            except StopIteration:
                self.current_action = None
        
        if self.current_action is None:
            simple_action = self.learner.sample_action(obs,
                next_opponent_action=predicted_opponent_action
            )
            self.current_action = ActionMap.simple_to_discrete(simple_action)
            action = next(self.current_action)
        
        return action

    def _obtain_effective_action(self, previous_move_index: int, current_move_index: int, previous_move_progress: float, current_move_progress: float) -> int | None:
        """Obtain what action was performed by the player. If the action was inefffectual, return `None`."""
        move = FOOTSIES_MOVE_INDEX_TO_MOVE[previous_move_index]
        actionable = ActionMap.is_state_actionable_late(move, previous_move_progress, current_move_progress)
        return ActionMap.simple_from_move_index(current_move_index) if actionable else None

    def update(self, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict):
        # NOTE: with the way frameskipping is being done we are updating Q-values throughout the duration of moves, not "skipping" them
        obs_agent_action = self._obtain_effective_action(self.current_info["p1_move"], info["p1_move"], self.current_observation[0, 32].item(), next_obs[0, 32].item())
        obs_opponent_action = self._obtain_effective_action(self.current_info["p2_move"], info["p2_move"], self.current_observation[0, 33].item(), next_obs[0, 33].item())
        
        self.learner.learn(self.current_observation, next_obs, reward, terminated, truncated,
            obs_agent_action=obs_agent_action,
            obs_opponent_action=obs_opponent_action,
        )
        
        self.cumulative_delta += self.learner.delta
        self.cumulative_delta_n += 1

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

    def save(self, folder_path: str):
        model_path = os.path.join(folder_path, "model")
        torch.save(self.model.state_dict(), model_path)

    def evaluate_average_delta(self) -> float:
        res = (
            self.cumulative_delta / self.cumulative_delta_n
        ) if self.cumulative_delta_n != 0 else 0

        self.cumulative_delta = 0
        self.cumulative_delta_n = 0

        return res

    def _initialize_test_states(self, test_states: list[torch.Tensor]):
        if self._test_observations is None:
            test_observations, _ = zip(test_states)
            self._test_observations = torch.tensor(test_observations, dtype=torch.float32)

    def evaluate_average_policy_entropy(self, test_states: list[tuple[Any, Any]]) -> float:
        self._initialize_test_states(test_states)

        with torch.no_grad():
            probs = self.actor(self._test_observations)
            entropies = -torch.sum(torch.log(probs + 1e-8) * probs)
            return torch.mean(entropies).item()
