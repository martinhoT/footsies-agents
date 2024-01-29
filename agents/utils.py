from copy import copy
from typing import Callable, Tuple
from gymnasium import Env, ObservationWrapper, ActionWrapper
from stable_baselines3.common.base_class import BaseAlgorithm

# Some wrappers need to be handled in a special manner when extracting a policy for the FOOTSIES environment
from footsies_gym.wrappers.frame_skip import FootsiesFrameSkipped
from footsies_gym.wrappers.normalization import FootsiesNormalized


def wrap_policy(
    env: Env, internal_policy: Callable
) -> Callable[[dict], Tuple[bool, bool, bool]]:
    observation_wrappers = []
    footsies_observation_wrappers = []  # these need to be applied before frameskipping
    action_wrappers = []

    frameskip_wrapper = None

    current_env = env
    while current_env != current_env.unwrapped:
        if isinstance(current_env, ObservationWrapper):
            if isinstance(current_env, FootsiesNormalized):
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
        # NOTE: it's assumed that the frameskip wrapper is wrapped by any other observation/action wrappers, except those for FOOTSIES
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


def snapshot_sb3_policy(agent: BaseAlgorithm, deterministic: bool = False):
    policy = copy(agent.policy)
    policy.load_state_dict(agent.policy.state_dict())

    def wrapper(obs):
        return policy.predict(obs, deterministic=deterministic)[0].item()

    return wrapper
