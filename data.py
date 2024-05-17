import numpy as np
import gzip
import os
import struct
import random
from torch.utils.data import Dataset
from typing import Any, Callable, Generator, Iterable, Iterator
from dataclasses import dataclass, field, astuple
from itertools import pairwise
from gymnasium import Env
from gymnasium.wrappers.flatten_observation import FlattenObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.utils import get_dict_obs_from_vector_obs
from agents.action import ActionMap
from io import BufferedIOBase


UNFLATTENED_OBSERVATION_SPACE = FootsiesNormalized(FootsiesEnv()).observation_space


@dataclass
class FootsiesTransition:
    obs:            np.ndarray
    next_obs:       np.ndarray
    reward:         float
    p1_action:      np.ndarray
    p2_action:      np.ndarray
    terminated:     bool

    @property
    def info(self) -> dict:
        return get_dict_obs_from_vector_obs(self.obs, unflattenend_observation_space=UNFLATTENED_OBSERVATION_SPACE)

    @property
    def next_info(self) -> dict:
        return get_dict_obs_from_vector_obs(self.next_obs, unflattenend_observation_space=UNFLATTENED_OBSERVATION_SPACE)


# repr is False to avoid printing the entire episode, which is especially important when debugging
@dataclass(slots=True, repr=False)
class FootsiesEpisode:
    """
    Data container of an episode in FOOTSIES, from beginning to end.
    Contains the `n` experienced observations, and the `n-1` player 1 and 2 actions.
    The final observation is terminal, on which no actions were performed.

    NOTE: it's assumed that the episode has terminated, and not truncated.
    """
    observations:       np.ndarray
    p1_actions:         np.ndarray
    p2_actions:         np.ndarray
    rewards:            np.ndarray

    @property
    def p1_won(self) -> bool:
        """Whether player 1 won"""
        # P2 was standing at the end, which signifies death
        return self.observations[-1, 17] == 1
    
    @property
    def p2_won(self) -> bool:
        """Whether player 2 won"""
        # P1 was standing at the end, which signifies death
        return self.observations[-1, 2] == 1

    @property
    def steps(self) -> int:
        """Number of steps in the episode, excluding the terminal state"""
        return self.observations.shape[0] - 1

    def tobytes(self) -> bytes:
        """
        Serialize episode into bytes.
        
        The episode's number of steps should not be longer than `2^32 - 1`.
        """

        if self.steps > 2**32 - 1:
            raise ValueError("episode is too long, should be no longer than 2^32 - 1 (is this even possible?)")
        
        observations = self.observations.tobytes(order="C")
        p1_actions = self.p1_actions.tobytes(order="C")
        p2_actions = self.p2_actions.tobytes(order="C")
        rewards = self.rewards.tobytes(order="C")

        # Episode length
        prefix = struct.pack("<I", self.steps)

        return prefix + observations + p1_actions + p2_actions + rewards

    @staticmethod
    def frombytes(b: bytes) -> tuple["FootsiesEpisode", int]:
        """Deserialize an episode from bytes. Returns the episode and the number of bytes read"""
        pointer = 0
        
        n_steps = struct.unpack("<I", b[pointer:pointer + 4])[0]
        pointer += 4

        observations = np.frombuffer(b[pointer:pointer + 8 * 36 * (n_steps + 1)], dtype=np.float64).reshape(-1, 36, order="C")
        pointer += 8 * 36 * (n_steps + 1)

        p1_actions = np.frombuffer(b[pointer:pointer + n_steps], dtype=np.int8).reshape(-1, 1, order="C")
        pointer += n_steps

        p2_actions = np.frombuffer(b[pointer:pointer + n_steps], dtype=np.int8).reshape(-1, 1, order="C")
        pointer += n_steps

        rewards = np.frombuffer(b[pointer:pointer + 4 * n_steps], dtype=np.float32).reshape(-1, 1, order="C")
        pointer += 4 * n_steps

        return FootsiesEpisode(
            observations=observations,
            p1_actions=p1_actions,
            p2_actions=p2_actions,
            rewards=rewards,
        ), pointer

    @staticmethod
    def fromfile(file: BufferedIOBase) -> "FootsiesEpisode | None":
        """Deserialize an episode from a file"""
        n_steps_bytes = file.read(4)
        if len(n_steps_bytes) == 0:
            return None

        n_steps = struct.unpack("<I", n_steps_bytes)[0]

        observations = np.frombuffer(file.read(8 * 36 * (n_steps + 1)), dtype=np.float64).reshape(-1, 36, order="C")
        p1_actions = np.frombuffer(file.read(n_steps), dtype=np.int8).reshape(-1, 1, order="C")
        p2_actions = np.frombuffer(file.read(n_steps), dtype=np.int8).reshape(-1, 1, order="C")
        rewards = np.frombuffer(file.read(4 * n_steps), dtype=np.float32).reshape(-1, 1, order="C")

        return FootsiesEpisode(
            observations=observations,
            p1_actions=p1_actions,
            p2_actions=p2_actions,
            rewards=rewards,
        )

    @staticmethod
    def trajectory(episode: Iterable[tuple[np.ndarray, float, dict]]) -> "FootsiesEpisode":
        """Instantiate an episode from a trajectory of observations and info dictionaries."""
        observations, rewards, infos = zip(*episode, strict=True)

        # Check that trajectory is well built
        if not all(i2["frame"] - i1["frame"] == 1 for i1, i2 in pairwise(infos)):
            raise ValueError("trajectory is malformed, frames were skipped")

        # Concatenate all observations into a single array
        observations = np.vstack(observations)

        # Convert rewards into an array. There is no reward on reset, so we discard it
        rewards = np.array(rewards[1:], dtype=np.float32).reshape(-1, 1)

        # Discard first info dict since it doesn't contain action data
        infos = infos[1:]

        # Extract actions from info dicts
        agent_actions = np.array([
            ActionMap.primitive_to_discrete(info["p1_action"]) for info in infos
        ], dtype=np.int8).reshape(-1, 1)
        opponent_actions = np.array([
            ActionMap.primitive_to_discrete(info["p2_action"]) for info in infos
        ], dtype=np.int8).reshape(-1, 1)

        return FootsiesEpisode(
            observations=observations,
            p1_actions=agent_actions,
            p2_actions=opponent_actions,
            rewards=rewards,
        )

    def __iter__(self) -> Iterator[FootsiesTransition]:
        for i, ((obs, next_obs), reward, p1_action, p2_action) in enumerate(zip(pairwise(self.observations), self.rewards, self.p1_actions, self.p2_actions)):
            terminated = i >= self.steps - 1
            yield FootsiesTransition(obs, next_obs, reward, p1_action, p2_action, terminated)
    
    def __getitem__(self, idx: int) -> FootsiesTransition:
        obs = self.observations[idx, :]
        next_obs = self.observations[idx + 1, :]
        reward = self.rewards[idx, 0]
        p1_action = self.p1_actions[idx, 0]
        p2_action = self.p2_actions[idx, 0]
        terminated = idx >= self.steps - 1
        return FootsiesTransition(obs, next_obs, reward, p1_action, p2_action, terminated)

    def __len__(self) -> int:
        return self.steps


@dataclass(slots=True, repr=False)
class FootsiesDataset:
    """Dataset of transitions on the FOOTSIES environment."""
    episodes:           list[FootsiesEpisode]

    transitions:        tuple[FootsiesTransition, ...] = field(default_factory=tuple, init=False)

    def __post_init__(self):
        self._update_transitions()

    def __add__(self, other: "FootsiesDataset"):
        return FootsiesDataset(self.episodes + other.episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx) -> FootsiesEpisode:
        return self.episodes[idx]

    def _update_transitions(self):
        self.transitions = tuple(transition for episode in self.episodes for transition in episode)

    @staticmethod
    def load(path: str) -> "FootsiesDataset":
        with gzip.open(path, "rb") as f:
            episodes = []
            while (episode := FootsiesEpisode.fromfile(f)) is not None:
                episodes.append(episode)
                f.read(1)  # Skip newline character
        
        return FootsiesDataset(episodes)

    def save(self, path: str):
        if os.path.exists(path):
            raise ValueError(f"a file already exists at '{path}', won't overwrite")

        with gzip.open(path, "wb") as f:
            for episode in self.episodes:
                f.write(episode.tobytes())
                f.write(b"\n")
    
    def shuffle(self):
        """Shuffle the episodes (not the transitions)."""
        random.shuffle(self.episodes)
        self._update_transitions()

    def generate_split(self, fraction: float) -> tuple["FootsiesDataset", "FootsiesDataset"]:
        """Split the dataset into two, according to episodes."""
        l = len(self.episodes)
        split_point = int(l * fraction)
        episodes_0 = self.episodes[:split_point]
        episodes_1 = self.episodes[split_point:]
        return FootsiesDataset(episodes_0), FootsiesDataset(episodes_1)

    def visualize(self, episode: int = 758):
        """Visualize an episode from the dataset."""
        e = self.episodes[episode]
        
        class FixedOpponent:
            def __init__(self, opponent_actions: list[int]):
                self.action_iterator = iter(opponent_actions)

            def __call__(self, obs: dict, info: dict) -> tuple[bool, bool, bool]:
                try:
                    action = next(self.action_iterator)
                except StopIteration:
                    action = 0

                t = ((action & 1) != 0), ((action & 2) != 0), ((action & 4) != 0)
                return t

        opponent = FixedOpponent(list(e.p2_actions.flatten()))

        footsies_env = FootsiesEnv(
            game_path="../Footsies-Gym/Build/FOOTSIES.x86_64",
            dense_reward=True,
            frame_delay=0,
            render_mode="human",
            skip_instancing=False,
            fast_forward=False,
            sync_mode="synced_non_blocking",
            by_example=False,
            vs_player=False,
            opponent=opponent,
            log_file="out.log",
            log_file_overwrite=True,
            **FootsiesEnv.find_ports(11000), # type: ignore
        )

        footsies_normalized = FootsiesNormalized(footsies_env)
        env = FlattenObservation(footsies_normalized)
        
        obs, _ = env.reset()

        # Load the initial observation that is in the dataset. This is because the bot could act the instant reset() was called, but that action wasn't saved
        battle_state = footsies_env.save_battle_state()
        initial_obs = get_dict_obs_from_vector_obs(e.observations[0, :], unflattenend_observation_space=footsies_normalized.observation_space)
        battle_state.p1State.position[0] = initial_obs["position"][0].item()
        battle_state.p2State.position[0] = initial_obs["position"][1].item()
        battle_state.p1State.guardHealth = initial_obs["guard"][0].item()
        battle_state.p2State.guardHealth = initial_obs["guard"][1].item()
        battle_state.p1State.currentActionID = initial_obs["move"][0].item()
        battle_state.p2State.currentActionID = initial_obs["move"][1].item()
        battle_state.p1State.currentActionFrame = initial_obs["move_frame"][0].item()
        battle_state.p2State.currentActionFrame = initial_obs["move_frame"][1].item()
        footsies_env.load_battle_state(battle_state)

        for recorded_obs, p1_action in zip(list(e.observations[1:, :]), list(e.p1_actions.flatten())):
            t = ((p1_action & 1) != 0), ((p1_action & 2) != 0), ((p1_action & 4) != 0)
            obs, _, _, _, _ = env.step(t)
            if not np.isclose(recorded_obs, obs).all():
                raise RuntimeError(f"the recorded dataset can't correctly reproduce the episode (the recorded observation {recorded_obs} doesn't match the actual observation {obs})")

        env.close()


class FootsiesTorchDataset(Dataset):
    """
    Flattened dataset of transitions, for usage with PyTorch's `DataLoader`.
    
    Returns tuples of (`obs`, `next_obs`, `reward`, `p1_action`, `p2_action`, `terminated`)
    """
    def __init__(self, dataset: FootsiesDataset):
        self._dataset = dataset

        self._distinct = False
        self._update_transitions(distinct=False)

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        return self._transitions[idx]

    def __len__(self) -> int:
        return len(self._transitions)

    @property
    def dataset(self) -> FootsiesDataset:
        return self._dataset

    @property
    def distinct(self) -> bool:
        return self._distinct

    @distinct.setter
    def distinct(self, value: bool):
        if self._distinct != value:
            self._update_transitions(distinct=value)
        self._distinct = value
    
    def _update_transitions(self, distinct: bool = False):
        if distinct:
            self._transitions = tuple(map(astuple, set(self.dataset.transitions)))
        else:
            self._transitions = tuple(map(astuple, self.dataset.transitions))


class DataCollector:
    def __init__(
        self,
        env: Env,
        action_source: Callable[[np.ndarray], Any] | None = None,
    ):
        """
        Class for collecting play data from the FOOTSIES environment.
        """
        if not isinstance(env.unwrapped, FootsiesEnv):
            raise ValueError("the underlying environment should be FOOTSIES (FootsiesEnv)")

        if action_source is None:
            action_source = lambda obs: env.action_space.sample()
        
        self.env = env
        self.action_source = action_source
    
    def sample_trajectory(self) -> Generator[tuple[np.ndarray, float, dict], None, None]:
        """
        Sample a trajectory from the environment.
        """
        obs, _ = self.env.reset()
        terminated, truncated = False, False,
        while not (terminated or truncated):
            action = self.action_source(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            yield obs, float(reward), info

    def collect(self, episodes: int) -> FootsiesDataset:
        episode_list = [
            FootsiesEpisode.trajectory(self.sample_trajectory())
            for _ in range(episodes)
        ]

        return FootsiesDataset(episode_list)
    
    def stop(self):
        self.env.close()

    @staticmethod
    def bot_vs_human(
        game_path: str = "../Footsies-Gym/Build/FOOTSIES.x86_64",
        dense_reward: bool = True,
        **kwargs,
    ) -> "DataCollector":
        """Instantiate a data collector that obtains human play data against the in-game bot."""
        footsies_env = FootsiesEnv(
            game_path=game_path,
            dense_reward=dense_reward,
            frame_delay=0,
            render_mode="human",
            skip_instancing=False,
            fast_forward=False,
            sync_mode="synced_non_blocking",
            by_example=True,
            vs_player=True,
            **kwargs,
        )

        env = FootsiesActionCombinationsDiscretized(
            FlattenObservation(
                FootsiesNormalized(
                    footsies_env
                )
            )
        )

        return DataCollector(
            env=env,
            action_source=None,
        )
