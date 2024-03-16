import psutil
from itertools import count
from collections import deque
from copy import copy
from typing import Any, Callable
from gymnasium import Env, ObservationWrapper, ActionWrapper, Wrapper
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from agents.action import ActionMap
from footsies_gym.moves import FootsiesMove
from torch.utils.tensorboard import SummaryWriter

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


def find_footsies_ports(start: int = 11000, step: int = 1, end: int = None) -> tuple[int, int, int]:
    closed_ports = {p.laddr.port for p in psutil.net_connections(kind="tcp4")}

    ports = []

    port_iterator = count(start=start, step=step) if end is None else range(start, end, step)

    for port in port_iterator:
        if port not in closed_ports:
            ports.append(port)

        if len(ports) >= 3:
            break

    if len(ports) < 3:
        raise RuntimeError(f"could not find 3 free ports for a new FOOTSIES instance (starting at {start} with steps of {step})")

    return tuple(ports)


class AppendSimpleHistoryWrapper(ObservationWrapper):
    """Observation wrapper for appending the action history of one of the players. Must be put after `FootsiesNormalized`."""
    def __init__(self, env, p1: bool, n: int, action_dim: int, distinct: bool):
        super().__init__(env)

        e = env
        has_normalized = False
        while e != e.unwrapped:
            if isinstance(env, FootsiesNormalized):
                has_normalized = True
                break
            e = e.env
        
        if not has_normalized:
            raise ValueError("AppendSimpleHistoryWrapper must be put after FootsiesNormalized")

        self.observation_space: spaces.Dict = env.observation_space
        self.observation_space.spaces[f"p{1 if p1 else 2}_history"] = spaces.MultiDiscrete([action_dim] * n)

        self.p1 = p1
        self.action_dim = action_dim
        self.distinct = distinct
        # Fill history with no-ops
        self.history = deque([0] * n, maxlen=n)

        self.prev_obs = None
    
    def observation(self, obs: dict) -> dict:
        obs = obs.copy()
        
        # Determine which action the player performed into `obs`
        if self.prev_obs is not None:
            action = ActionMap.simple_from_transition(
                previous_player_move_index=self.prev_obs["move"][0 if self.p1 else 1],
                previous_opponent_move_index=self.prev_obs["move"][1 if self.p1 else 0],
                previous_player_move_progress=self.prev_obs["move_frame"][0 if self.p1 else 1],
                previous_opponent_move_progress=self.prev_obs["move_frame"][1 if self.p1 else 0],
                current_player_move_index=obs["move"][0 if self.p1 else 1],
            )

            if not self.distinct or (self.history[-1] != action):
                self.history.append(action)

        self.prev_obs = obs

        # Append the history of actions so far to the observation
        obs[f"p{1 if self.p1 else 2}_history"] = list(self.history)

        return obs
    

class FootsiesEncourageAdvance(Wrapper):
    """Wrapper for providing a reward for advancing towards the opponent based on the distance between players. Must be put after `FootsiesNormalized`."""
    
    def __init__(self, env,
        distance_cap: float = 4.0,
        advance_reward: float = 0.01,
        log_dir: str = None,
    ):
        """
        Wrapper for providing a reward for advancing towards the opponent based on the distance between players.
        
        Parameters
        ----------
        - `distance_cap`: the maximum distance beyond which 0 reward will be given. The default corresponds to round start position
        - `advance_reward`: the maximum reward for advancing towards the opponent, which will decrease towards 0 the greater the distance is between the players
        - `log_dir`: the directory where to save the Tensorboard logs regarding how much extra reward was given on each episode
        """
        super().__init__(env)

        self.distance_cap = distance_cap
        self.advance_reward = advance_reward

        # For logging
        self._summary_writer = SummaryWriter()
        self._episode_extra_reward = 0.0
        self._episodes = 0

    def step(self, action: tuple[bool, bool, bool]) -> tuple[dict, float, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        distance_between_players = min(obs["position"][1] - obs["position"][0], self.distance_cap)

        # Reward agent for advancing forward, and close to the opponent
        p1_move = ActionMap.move_from_move_index(info["p1_move"])
        if p1_move == FootsiesMove.FORWARD:
            extra = self.advance_reward * (self.distance_cap - distance_between_players) / self.distance_cap
            self._episode_extra_reward += extra
            reward += extra

        # Logging
        if terminated or truncated:
            if self._summary_writer is not None:
                self._summary_writer.add_scalar(
                    "Performance/Reward from advancing",
                    self._episode_extra_reward,
                    self._episodes,
                )

            self._episode_extra_reward = 0.0
            self._episodes += 1

        return obs, reward, terminated, truncated, info
