from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple
from gymnasium import Env

# Somw wrappers need to be handled in a special manner when extracting a policy for the FOOTSIES environment
from footsies_gym.wrappers.frame_skip import FootsiesFrameSkipped
from footsies_gym.wrappers.normalization import FootsiesNormalized
from gymnasium import ObservationWrapper, ActionWrapper


class FootsiesAgentBase(ABC):
    @abstractmethod
    def act(self, obs) -> Any:
        """Get the chosen action for the currently observed environment state"""

    @abstractmethod
    def update(self, next_obs, reward: float, terminated: bool, truncated: bool, info: dict):
        """After acting in the environment, process the perceived observation and reward"""

    def preprocess(self, env: Env):
        """Do some preprocessing on the environment before training on it"""
        pass

    @abstractmethod
    def load(self, folder_path: str):
        """Load the agent from disk"""

    @abstractmethod
    def save(self, folder_path: str):
        """Save the agent to disk (overwriting an already saved agent at that path)"""

    @abstractmethod
    def extract_policy(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        """
        Extract a policy which can be provided to the FOOTSIES environment as an opponent.

        Subclasses of `FootsiesAgentBase` should implement the `extract_policy` method themselves.
        This method should return `super()._extract_policy(env, wrapped)`, where `wrapped` is the subclass's policy
        as used for training.
        This is relevant since the environment can be wrapped with observation and action wrappers, of which the
        subclass agent is not aware (or at least doesn't need to be).
        The superclass method deals with some of these wrappers to avoid code duplication
        """

    def _extract_policy(
        self, env: Env, internal_policy: Callable
    ) -> Callable[[dict], Tuple[bool, bool, bool]]:
        observation_wrappers = []
        footsies_observation_wrappers = (
            []
        )  # these need to be applied before frameskipping
        action_wrappers = []

        frameskip_wrapper = None

        current_env = env
        while current_env != current_env.unwrapped:
            if isinstance(current_env, ObservationWrapper):
                if isinstance(current_env, (FootsiesNormalized,)):
                    footsies_observation_wrappers.append(current_env)
                else:
                    observation_wrappers.append(current_env)

            elif isinstance(current_env, ActionWrapper):
                action_wrappers.append(current_env)

            elif isinstance(current_env, FootsiesFrameSkipped):
                frameskip_wrapper = current_env

            current_env = current_env.env

        def policy(obs: dict) -> Tuple[bool, bool, bool]:
            for footsies_observation_wrapper in reversed(footsies_observation_wrappers):
                obs = footsies_observation_wrapper.observation(obs)

            # TODO: not the best solution, the condition is always evaluated even though it has always the same value
            # NOTE: it's assumed that the frameskip wrapper is below any other observation/action wrappers
            if frameskip_wrapper is not None:
                if frameskip_wrapper._is_obs_skippable(obs):
                    return (False, False, False)

                obs = frameskip_wrapper._frame_skip_obs(obs)

            for observation_wrapper in reversed(observation_wrappers):
                obs = observation_wrapper.observation(obs)

            action = internal_policy(obs)

            for action_wrapper in action_wrappers:
                action = action_wrapper.action(action)

            return action

        return policy
