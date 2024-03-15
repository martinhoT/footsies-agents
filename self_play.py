import random
import logging
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from dataclasses import dataclass
from typing import Callable

LOGGER = logging.getLogger("main.self_play")


@dataclass
class Opponent:
    method:     Callable[..., "any"]
    created_at: int
    elo:        int         = 1200

    def __str__(self):
        return f"Opponent(created_at={self.created_at}, elo={self.elo})"


class SelfPlayManager:
    """
    Class that manages an opponent pool for self-play.
    
    Opponents are sampled uniformly from the pool (FSP).
    We could give more weight to latest agent snapshots (SP), but that can generate cycles in the found strategies.
    Instead, it would be better to give preference to opponents that the agent has trouble beating (PFSP).
    However, only FSP is used in this implementation.
    Check the 'Prioritized fictitious self-play' section of the StarCraft II paper for more details on a possible PFSP implementation.
    """

    def __init__(
        self,
        snapshot_method: callable,
        max_snapshots: int = 10,
        snapshot_interval: int = 2000,
        switch_interval: int = 100,
        mix_bot: int = 1,
        log_elo: bool = False,
        log_dir: str = None,
        log_interval: int = 1,
        starter_opponent: Callable[[dict], tuple[bool, bool, bool]] = None,
    ):
        """
        Instance of an opponent pool manager for self-play.
        
        Parameters
        ----------
        - `snapshot_method`: producer of agent snapshots to serve as future opponents
        - `max_snapshots`: maximum capacity of the opponent pool. If at maximum, the oldest opponents are discarded
        - `snapshot_interval`: the interval between snapshots of the current policy for the opponent pool, in number of episodes
        - `switch_interval`: the interval between opponent switched, in number of episodes
        - `mix_bot`: how many opponents will the in-game opponent count as, when sampling from the opponent pool.
        For example, if `mix_bot` is equal to `max_snapshots // 2`, then the in-game bot will be sampled half of the time, when compared with the snapshots.
        If 0, then the in-game bot will never be used
        - `log_elo`: whether to log the ELO of the agent
        - `log_dir`: the directory where to save the self-play logs using Tensorboard. If `None`, then no logs are saved
        - `log_interval`: the interval between log writes, in number of episodes
        - `starter_opponent`: the opponent to start with, should ideally be the one passed to the environment. If `None`, then the in-game bot is used, and will be assumed to be the one used by the environment
        """
        self.snapshot_method = snapshot_method
        self.max_snapshots = max_snapshots
        self.snapshot_interval = snapshot_interval
        self.switch_interval = switch_interval
        self.mix_bot = mix_bot

        self.opponent_pool: deque[Opponent] = deque([], maxlen=max_snapshots)
        if starter_opponent is not None:
            self._register_opponent(starter_opponent, episode=0)
        self.episode = 0

        ingame_bot = Opponent(None, -1)
        self._ingame_bot = ingame_bot
        self._current_opponent = self.opponent_pool[0] if starter_opponent is not None else ingame_bot
        self._agent_elo = 1200
        self._elo_k = 32

        # Logging
        self.summary_writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None
        self.log_elo = log_elo
        self.log_frequency = log_interval

    def _register_opponent(self, opponent_f: Callable[..., "any"], episode: int = None) -> Opponent:
        if episode is None:
            episode = self.episode

        opponent = Opponent(opponent_f, episode)
        self.opponent_pool.append(opponent)
        return opponent

    def _sample_opponent(self) -> Opponent:
        full_pool = [self._ingame_bot] + list(self.opponent_pool)
        counts = [self.mix_bot] + [1] * len(self.opponent_pool)
        return random.sample(full_pool, counts=counts, k=1)[0]

    def update_at_episode(self, game_result: int) -> bool:
        """
        Update the manager after every episode. Returns whether an opponent change was done.
        
        Parameters
        ----------
        - `game_result`: the result of the game, in terms of the agent's perspective. 1 for win, 0 for loss, 0.5 for draw
        """
        previous_opponent = self._current_opponent
        self.episode += 1

        # Update the agent's and opponent's ELO
        expected_agent_victory = 1 / (1 + 10 ** ((self._current_opponent.elo - self._agent_elo) / 400))
        expected_opponent_victory = 1 - expected_agent_victory
        self._agent_elo += self._elo_k * (game_result - expected_agent_victory)
        self._current_opponent.elo += self._elo_k * ((1 - game_result) - expected_opponent_victory)

        # Perform a snapshot of the agent at the current
        if self.episode % self.snapshot_interval == 0:
            new_opponent = self._register_opponent(self.snapshot_method())
            LOGGER.info("Agent snapshot created! (new opponent created at episode %s, with ELO %s)", new_opponent.created_at, new_opponent.elo)
        
        # Switch to another opponent
        if self.episode % self.switch_interval == 0:
            self._current_opponent = self._sample_opponent()
            LOGGER.info("Switched to a new opponent! (the one created at episode %s, with ELO %s)", self._current_opponent.created_at, self._current_opponent.elo)
            league_info = '\n'.join(map(str, [self._ingame_bot] + list(self.opponent_pool)))
            LOGGER.info("League: %s", league_info)

        # Logging
        if self.summary_writer is not None:
            if self.log_elo and self.episode % self.log_frequency == 0:
                self.summary_writer.add_scalar(
                    "Performance/ELO",
                    self.elo,
                    self.episode,
                )

        return self._current_opponent != previous_opponent

    @property
    def current_opponent(self) -> Opponent:
        """The opponent being currently used for self-play. If `None`, then the in-game bot is being used."""
        return self._current_opponent
    
    @property
    def elo(self) -> int:
        """The ELO of the agent, calculated according to the ELO of the opponents."""
        return self._agent_elo