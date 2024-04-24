from copy import copy
from typing import Any, Callable
from gymnasium import Env, ObservationWrapper, ActionWrapper
from stable_baselines3.common.base_class import BaseAlgorithm
from agents.wrappers import FootsiesSimpleActions, FootsiesSimpleActionExecutor


def wrap_policy(
    env: Env, internal_policy: Callable
) -> Callable[[dict, dict], tuple[bool, bool, bool]]:
    observation_wrappers = []
    action_wrappers = []

    # We treat the simple actions wrapper differently.
    # This wrapper would ideally be an action wrapper but oh well.
    using_simple_actions_wrapper: bool = False

    current_env = env
    while current_env != current_env.unwrapped:
        if isinstance(current_env, ObservationWrapper):
            observation_wrappers.append(current_env)

        elif isinstance(current_env, ActionWrapper):
            action_wrappers.append(current_env)

        elif isinstance(current_env, FootsiesSimpleActions):
            using_simple_actions_wrapper = True

        current_env = current_env.env # type: ignore

    simple_action_executor = FootsiesSimpleActionExecutor(True) if using_simple_actions_wrapper else None

    def policy(obs: dict, info: dict) -> tuple[bool, bool, bool]:
        for observation_wrapper in reversed(observation_wrappers):
            obs = observation_wrapper.observation(obs)

        action = internal_policy(obs, info)

        if simple_action_executor is not None:
            action = simple_action_executor.act(action)

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


def extract_sub_kwargs(kwargs: dict, subkeys: tuple[str], strict: bool = True) -> list[dict[str, Any]]:
    """
    Extract keyword arguments from `kwargs` with the provided `subkeys`. If `strict`, will raise an error in case not all keys in `kwargs` were exhausted
    
    NOTE: multiple periods `'.'` are supported in subkey names, with the first period delimiting the subkey name

    NOTE: shouldn't be used anymore. This was used to specify arguments through the command line, but it's better to just specify initialization in the models folder

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


def observation_invert_perspective(obs: dict) -> dict:
    """Invert the observation's perspective."""
    inverted = obs.copy()
    
    inverted["guard"] = tuple(reversed(inverted["guard"]))
    inverted["move"] = tuple(reversed(inverted["move"]))
    inverted["move_frame"] = tuple(reversed(inverted["move_frame"]))
    inverted["position"] = tuple(map(float.__neg__, reversed(inverted["position"])))

    return inverted
