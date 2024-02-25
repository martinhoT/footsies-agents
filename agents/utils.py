from copy import copy
from typing import Any, Callable, Iterable
from gymnasium import Env, ObservationWrapper, ActionWrapper
from stable_baselines3.common.base_class import BaseAlgorithm

# Some wrappers need to be handled in a special manner when extracting a policy for the FOOTSIES environment
from footsies_gym.wrappers.frame_skip import FootsiesFrameSkipped
from footsies_gym.wrappers.normalization import FootsiesNormalized


def wrap_policy(
    env: Env, internal_policy: Callable
) -> Callable[[dict], tuple[bool, bool, bool]]:
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

    def policy(obs: dict) -> tuple[bool, bool, bool]:
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


def extract_sub_kwargs(kwargs: dict, subkeys: tuple[str], strict: bool = True) -> tuple[dict[str, Any]]:
    """
    Extract keyword arguments from `kwargs` with the provided `subkeys`. If `strict`, will raise an error in case not all keys in `kwargs` were exhausted
    
    NOTE: multiple periods `'.'` are supported in subkey names, with the first period delimiting the subkey name

    Example:
    ```
    >>> kwargs = {
    ...     "m1.a": 1,
    ...     "m1.b": 2,
    ...     "m2.a": 3,
    ... }
    >>> keys = ["m1", "m2"]
    >>> extract_sub_kwargs(kwargs, keys, strict=True)
    [{"a": 1, "b": 2}, {"a": 3}]
    ```
    """
    if strict:
        unknown_kwargs = {k for k in kwargs.keys() if (".") not in k}
        if unknown_kwargs:
            raise ValueError(f"since it's strict, all keyword arguments must be prefixed with a subkey name ('<subkey>.<kwarg>'), unknown: {unknown_kwargs}")
        subkeys_set = set(subkeys)
        unknown_subkeys = {sub for sub in map(lambda k: k.split(".")[0], kwargs.keys()) if sub not in subkeys_set}
        if unknown_subkeys:
            raise ValueError(f"since it's strict, all keyword arguments must be prefixed with one of subkeys {subkeys}, unknown: {unknown_subkeys}")

    extracted = [
        {".".join(k.split(".")[1:]): v for k, v in kwargs.items() if k.startswith(subkey + ".")}
        for subkey in subkeys
    ]

    return extracted
