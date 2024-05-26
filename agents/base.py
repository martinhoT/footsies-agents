from abc import ABC, abstractmethod
from typing import Any
from gymnasium import ActionWrapper, Env, ObservationWrapper
from torch import nn
from agents.utils import observation_invert_perspective
from agents.wrappers import FootsiesSimpleActionExecutor, FootsiesSimpleActionExtractor, FootsiesSimpleActions
from agents.action import ActionMap
from opponents.base import Opponent


class FootsiesAgentBase(ABC):
    @abstractmethod
    def act(self, obs, info: dict) -> Any:
        """Get the chosen action for the currently observed environment state."""

    @abstractmethod
    def update(self, obs, next_obs, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict):
        """Update the agent from an environment transtition."""

    def preprocess(self, env: Env):
        """Do some preprocessing on the environment before training on it."""

    @abstractmethod
    def load(self, folder_path: str):
        """Load the agent from disk"""

    @abstractmethod
    def save(self, folder_path: str):
        """Save the agent to disk (overwriting an already saved agent at that path)."""

    def extract_opponent(self, env: Env) -> "FootsiesAgentOpponent":
        """
        Extract a policy which can be provided to the FOOTSIES environment as an opponent.

        This method should ideally return a deep copy of the agent, to avoid state interference.

        NOTE: the internal policy implementations don't need to perform some transformations on the observations and actions
        to account for the change in perspective between player 1 (agent) and player 2 (opponent, i.e. the extracted policy).
        This is already handled externally.
        """
        raise NotImplementedError("this agent did not implement a way for extracting themselves")

    def reset(self):
        """Reset any internal state that the agent builds up over an episode. Important for when an opponent is extracted, so that we can reset its internal state."""


class FootsiesAgentTorch(FootsiesAgentBase):
    
    @property
    @abstractmethod
    def shareable_model(self) -> nn.Module:
        """The PyTorch module that can be shared with other agents during Hogwild! parallel training. Returns the `model` property by default."""


class FootsiesAgentOpponent(Opponent):
    def __init__(self, agent: FootsiesAgentBase, env: Env):
        self._agent = agent
        self._agent.reset()

        self._observation_wrappers: list[ObservationWrapper] = []
        self._action_wrappers: list[ActionWrapper] = []

        # We treat the simple actions wrapper differently.
        # This wrapper would ideally be an action wrapper but oh well.
        simple_actions_wrapper: FootsiesSimpleActions | None = None

        current_env = env
        while current_env != current_env.unwrapped:
            if isinstance(current_env, ObservationWrapper):
                self._observation_wrappers.append(current_env)

            elif isinstance(current_env, ActionWrapper):
                self._action_wrappers.append(current_env)

            elif isinstance(current_env, FootsiesSimpleActions):
                simple_actions_wrapper = current_env

            current_env = current_env.env # type: ignore

        if simple_actions_wrapper:
            self._simple_action_executor = FootsiesSimpleActionExecutor(
                allow_special_moves=simple_actions_wrapper.agent_allow_special_moves
            )
            self._simple_action_extractor = FootsiesSimpleActionExtractor(
                assumed_agent_action_on_nonactionable=simple_actions_wrapper.assumed_agent_action_on_nonactionable,
                assumed_opponent_action_on_nonactionable=simple_actions_wrapper.assumed_opponent_action_on_nonactionable,
            )
        else:
            self._simple_action_executor = None
            self._simple_action_extractor = None
        self._simple_action_stuff_should_reset = True

    def act(self, obs: dict, info: dict) -> tuple[bool, bool, bool]:
        # Invert the perspective, since the agent was trained as if they were on the left side of the screen
        obs = observation_invert_perspective(obs)
        info = observation_invert_perspective(info)
        
        if self._simple_action_extractor is not None and self._simple_action_executor is not None:
            if self._simple_action_stuff_should_reset:
                info = self._simple_action_executor.reset(info)
                info = self._simple_action_extractor.reset(info)
                self._simple_action_stuff_should_reset = False
            else:
                info = self._simple_action_executor.update(info)
                info = self._simple_action_extractor.update(info)

        for observation_wrapper in reversed(self._observation_wrappers):
            obs = observation_wrapper.observation(obs)
        
        action = self._agent.act(obs, info)

        # NOTE: this does not match how the executer is handled by the simple actions wrapper,
        #       since action is None in case the player cannot act.
        if self._simple_action_executor is not None:
            action = self._simple_action_executor.act(action)
        
        for action_wrapper in self._action_wrappers:
            action = action_wrapper.action(action)
        
        # Invert the action, again for the same reason as the observation
        action = ActionMap.invert_primitive(action)

        return action

    def reset(self):
        self._agent.reset()
        self._simple_action_stuff_should_reset = True

    @property
    def agent(self) -> FootsiesAgentBase:
        """The internal agent that is acting as the opponent."""
        return self._agent
    