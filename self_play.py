import random
import logging
from torch.utils.tensorboard import SummaryWriter
from collections import deque

LOGGER = logging.getLogger("main.self_play")


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
        """
        
        self.snapshot_method = snapshot_method
        self.max_snapshots = max_snapshots
        self.snapshot_interval = snapshot_interval
        self.switch_interval = switch_interval
        self.mix_bot = mix_bot

        self.opponent_pool = deque([], maxlen=max_snapshots)

        self.episode = 0

        self._current_opponent = None
        self._opponent_elos = {None: 1200} # ELO of the in-game bot
        self._agent_elo = 1200
        self._elo_k = 32

        # Logging
        self.summary_writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None
        self.log_elo = log_elo
        self.log_frequency = log_interval

    def _add_opponent(self, opponent):
        if len(self.opponent_pool) == self.opponent_pool.maxlen:
            self._opponent_elos.pop(self.opponent_pool[0])
        
        self.opponent_pool.append(opponent)
        self._opponent_elos[opponent] = 1200

    def _sample_opponent(self) -> "any":
        full_pool = [None] + self.opponent_pool
        counts = [self.mix_bot] + [1] * len(self.opponent_pool)
        return random.sample(full_pool, counts=counts, k=1)[0]

    def update_at_episode(self, game_result: int) -> bool:
        """
        Update the manager after every episode. Returns whether an opponent change was done.
        
        Parameters
        ----------
        - `game_result`: the result of the game, in terms of the agent's perspective. 1 for win, 0 for draw, -1 for loss
        """
        self.episode += 1
        previous_opponent = self._current_opponent

        # Update the agent's and opponent's ELO
        expected_agent_victory = 1 / (1 + 10 ** ((self._opponent_elos[self._current_opponent] - self._agent_elo) / 400))
        expected_opponent_victory = 1 - expected_agent_victory
        self._agent_elo += self._elo_k * (game_result - expected_agent_victory)
        self._opponent_elos[self._current_opponent] += self._elo_k * ((-game_result) - expected_opponent_victory)

        # Perform a snapshot of the agent at the current
        if self.episode % self.snapshot_interval == 0:
            self._add_opponent(self.snapshot_method())
            LOGGER.info("Agent snapshot created!")
        
        # Switch to another opponent
        if self.episode & self.switch_interval == 0:
            self._current_opponent = self._sample_opponent()
            LOGGER.info("Switched to a new opponent!")

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
    def current_opponent(self) -> callable:
        """The opponent being currently used for self-play. If `None`, then the in-game bot is being used."""
        return self._current_opponent
    
    @property
    def elo(self) -> int:
        """The ELO of the agent, calculated according to the ELO of the opponents."""
        return self._agent_elo