import numpy as np
import gzip
import os
import struct
from typing import Any, Callable, Generator, Iterable
from dataclasses import dataclass
from itertools import pairwise
from gymnasium import Env
from gymnasium.wrappers import FlattenObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from agents.action import ActionMap


@dataclass(slots=True)
class FootsiesEpisode:
    """
    Data container of an episode in FOOTSIES, from beginning to end.
    Contains the `n` experienced observations, and the `n-1` player 1 and 2 actions.
    The final observation is terminal, on which no actions were performed.
    """
    observations:       np.ndarray
    p1_actions:         np.ndarray
    p2_actions:         np.ndarray
    rewards:            np.ndarray

    @property
    def p1_won(self) -> bool:
        """Whether player 1 won"""
        # P2's health at the end is 0
        return self.observations[-1, 1] == 0
    
    @property
    def p2_won(self) -> bool:
        """Whether player 2 won"""
        # P1's health at the end is 0
        return self.observations[-1, 0] == 0

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
        
        states = self.observations.tobytes(order="C")
        p1_actions = self.p1_actions.tobytes(order="C")
        p2_actions = self.p2_actions.tobytes(order="C")
        rewards = self.rewards.tobytes(order="C")

        # Episode length
        prefix = struct.pack("<I", self.steps)

        return prefix + states + p1_actions + p2_actions + rewards

    @staticmethod
    def frombytes(b: bytes) -> "FootsiesEpisode":
        """Deserialize an episode from bytes."""
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


@dataclass(slots=True)
class FootsiesDataset:
    """
    Dataset of transitions on the FOOTSIES environment.
    """
    episodes:           list[FootsiesEpisode]

    def __add__(self, other: "FootsiesDataset"):
        return FootsiesDataset(self.episodes + other.episodes)

    def append(self, episode: FootsiesEpisode):
        self.episodes.append(episode)

    @staticmethod
    def load(path: str) -> "FootsiesDataset":
        with gzip.open(path, "rb") as f:
            episodes = [FootsiesEpisode.frombytes(episode_bytes.strip()) for episode_bytes in f]
        
        return FootsiesDataset(episodes)

    def save(self, path: str):
        if os.path.exists(path):
            raise ValueError(f"a file already exists at '{path}', won't overwrite")

        with gzip.open(path, "wb") as f:
            for episode in self.episodes:
                f.write(episode.tobytes())
                f.write(b"\n")


class DataCollector:
    def __init__(
        self,
        env: Env,
        action_source: Callable[[np.ndarray], Any] = None,
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
    
    def sample_trajectory(self) -> Generator[tuple[np.ndarray, dict], None, None]:
        """
        Sample a trajectory from the environment.
        """
        obs = self.env.reset()
        terminated, truncated = False, False,
        while not (terminated or truncated):
            action = self.action_source(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            yield obs, reward, info

    def collect(self, episodes: int) -> FootsiesDataset:
        episodes = [
            FootsiesEpisode.trajectory(self.sample_trajectory())
            for _ in range(episodes)
        ]

        return FootsiesDataset(episodes)
    
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
