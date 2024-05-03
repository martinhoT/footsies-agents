import random
import logging
from torch.utils.tensorboard import SummaryWriter # type: ignore
from collections import deque
from dataclasses import dataclass
from typing import Callable, cast
from opponents.base import OpponentManager, Opponent
from opponents.curriculum import CurriculumOpponent
from agents.base import FootsiesAgentOpponent, FootsiesAgentBase
from gymnasium import Env
from os import path
from footsies_gym.envs.footsies import FootsiesEnv
from tqdm import tqdm
import csv


LOGGER = logging.getLogger("main.self_play")


@dataclass
class SelfPlayOpponentInfo:
    opp:        FootsiesAgentOpponent | CurriculumOpponent | None
    name:       str
    elo:        float = 1200

    def __str__(self):
        return f"{self.name} (Elo: {self.elo})"


SnapshotMethod = Callable[[], FootsiesAgentOpponent]


class SelfPlayManager(OpponentManager):
    """
    Class that manages an opponent pool for self-play.
    
    Opponents are sampled uniformly from the pool (FSP).
    We could give more weight to latest agent snapshots (SP), but that can generate cycles in the found strategies.
    Instead, it would be better to give preference to opponents that the agent has trouble beating (PFSP).
    However, only FSP is used in this implementation.
    Check the 'Prioritized fictitious self-play' section of the StarCraft II paper for more details on a possible PFSP implementation.
    """

    BASE_ELO = 1200

    def __init__(
        self,
        agent: FootsiesAgentBase | None,
        env: Env,
        max_opponents: int = 10,
        snapshot_interval: int = 2000,
        switch_interval: int = 100,
        mix_bot: int = 1,
        win_rate_threshold: float = 0.8,
        starter_opponent: FootsiesAgentOpponent | None = None,
        evaluate_every: int | None = None,
        log_dir: str | None = None,
        csv_save: bool = True,
    ):
        """
        Instance of an opponent pool manager for self-play.
        
        Parameters
        ----------
        - `agent`: the agent, from which snapshots will be taken. May be `None` on initialization, but it *must* be set later before interacting with the environment
        - `env`: the environment on which evaluation will be performed, and from which snapshots are taken. Should *not* be wrapped by an opponent manager wrapper
        - `max_opponents`: maximum capacity of the opponent pool. If at maximum, the oldest opponents are discarded. The in-game bot is always present in the pool, and doesn't count towards this limit
        - `snapshot_interval`: the interval between snapshots of the current policy for the opponent pool, in number of episodes
        - `switch_interval`: the interval between opponent switched, in number of episodes
        - `mix_bot`: how many opponents will the in-game opponent count as, when sampling from the opponent pool.
        For example, if `mix_bot` is equal to `max_snapshots // 2`, then the in-game bot will be sampled half of the time, when compared with the snapshots.
        If 0, then the in-game bot will never be used
        - `win_rate_threshold`: the maximum win rate the agent can have against an opponent. If surpassed, then the agent switches to another opponent
        - `log_dir`: the directory where to save the self-play logs using Tensorboard. If `None`, then no logs are saved
        - `starter_opponent`: the opponent to start with, should ideally be the one passed to the environment. If `None`, then the in-game bot is used, and will be assumed to be the one used by the environment
        - `evaluate_every`: the interval in episodes with which to evaluate the agent against *all* past opponents. If `None`, no evaluation is performed
        """
        self.agent = agent
        self.env = env
        self.footsies_env = cast(FootsiesEnv, env.unwrapped)
        self.max_opponents = max_opponents
        self.snapshot_interval = snapshot_interval
        self.switch_interval = switch_interval
        self.mix_bot = mix_bot
        self.win_rate_threshold = win_rate_threshold
        self.evaluate_every = evaluate_every

        self.opponent_pool: deque[SelfPlayOpponentInfo] = deque([], maxlen=max_opponents)
        self.opponent_history: list[SelfPlayOpponentInfo] = [] # the opponent history explicitly includes the in-game bot
        if starter_opponent is not None:
            self._register_opponent(starter_opponent, "Starter opponent")
        self.episode = 0

        ingame_bot = SelfPlayOpponentInfo(None, "In-game bot")
        self._ingame_bot = ingame_bot
        self._current_opponent = self.opponent_pool[0] if starter_opponent is not None else ingame_bot
        self._agent_elo = self.BASE_ELO
        self._elo_k = 32
        self.opponent_history.append(ingame_bot)

        # Track the wins that the agent had against the current opponent.
        # Only games for the most recent switch are considered.
        # This is used to deetermine whether the agent has surpassed the current opponent and should switch to another one.
        self._current_wins = 0
        self._current_n_games = 0
        self._min_games = 20

        # Logging
        self.summary_writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None
        if csv_save and log_dir is not None:
            self._csv_file = open(path.join(log_dir, "performanceelo.csv"), mode="wt")
            self._csv_file_writer = csv.writer(self._csv_file, "unix", quoting=csv.QUOTE_MINIMAL)
        else:
            self._csv_file = None
            self._csv_file_writer = None

    def _register_opponent(self, opp: FootsiesAgentOpponent | CurriculumOpponent, name: str, elo: float = BASE_ELO) -> SelfPlayOpponentInfo:
        opponent = SelfPlayOpponentInfo(opp=opp, name=name, elo=elo)
        self.opponent_pool.append(opponent)
        self.opponent_history.append(opponent)
        return opponent

    def _sample_opponent(self) -> SelfPlayOpponentInfo:
        full_pool = [self._ingame_bot] + list(self.opponent_pool)
        counts = [self.mix_bot] + [1] * len(self.opponent_pool)
        return random.sample(full_pool, counts=counts, k=1)[0]

    def update_elo(self, opponent: SelfPlayOpponentInfo, game_result: float):
        """Update the agent's and opponent's Elo."""
        expected_agent_victory = 1 / (1 + 10 ** ((opponent.elo - self._agent_elo) / 400))
        expected_opponent_victory = 1 - expected_agent_victory
        self._agent_elo += self._elo_k * (game_result - expected_agent_victory)
        opponent.elo += self._elo_k * ((1 - game_result) - expected_opponent_victory)

    def update_at_episode(self, game_result: float) -> bool:
        """
        Update the manager after every episode. Returns whether an opponent change was done.
        
        Parameters
        ----------
        - `game_result`: the result of the game, in terms of the agent's perspective. 1 for win, 0 for loss, 0.5 for draw
        """
        if self.agent is None:
            raise RuntimeError("the snapshot method should have already been set before interacting with the environment, quitting")
        
        previous_opponent = self._current_opponent
        self.episode += 1
        self._current_wins += (game_result == 1.0)
        self._current_n_games += 1

        self.update_elo(self._current_opponent, game_result)

        # Perform a snapshot of the agent at the current
        if self.episode % self.snapshot_interval == 0:
            new_opponent = self._register_opponent(self.agent.extract_opponent(self.env), name=f"Snapshot at episode {self.episode}", elo=self.elo)
            LOGGER.info("Agent snapshot created at episode %s! (%s)", self.episode, new_opponent)
        
        # Switch to another opponent
        surpassed_opponent = (self._current_wins / self._current_n_games) > self.win_rate_threshold and self._current_n_games >= self._min_games
        if self.episode % self.switch_interval == 0 or surpassed_opponent:
            self._current_opponent = self._sample_opponent()
            LOGGER.info("Switched to a new opponent at episode %s! (%s)", self.episode, self._current_opponent)
            league_info = "\n".join(map(str, [self._ingame_bot] + list(self.opponent_pool)))
            league_info += f"\nAgent (Elo: {self.elo})"
            LOGGER.info("League: %s", league_info)

            self._current_n_games = 0
            self._current_wins = 0

        # Perform thorough evaluation of the agent's (and opponents') Elo
        if self.evaluate_every is not None and self.episode % self.evaluate_every == 0:
            self.update_elo_sweep()

        # Logging
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                "Performance/ELO",
                self.elo,
                self.episode,
            )
        if self._csv_file_writer is not None:
            self._csv_file_writer.writerow((self.episode, self.elo))

        return self._current_opponent != previous_opponent

    @property
    def current_opponent(self) -> Opponent | None:
        """The opponent being currently used for self-play. If `None`, then the in-game bot is being used."""
        return self._current_opponent.opp
    
    @property
    def elo(self) -> int:
        """The ELO of the agent, calculated according to the ELO of the opponents."""
        return self._agent_elo
    
    @property
    def exhausted(self) -> bool:
        # It's never exhausted
        return False

    def populate_with_curriculum_opponents(self, *opponents: CurriculumOpponent):
        """Populate the opponent pool with the given custom opponents pre-made for curriculum learning."""
        for opponent in opponents:
            self._register_opponent(opponent, name=opponent.__class__.__name__)

    def set_agent(self, agent: FootsiesAgentBase):
        """Set the agent, from which snapshots will be taken."""
        self.agent = agent

    def play_game_against(self, opponent: Opponent | None) -> int:
        """Have the agent play a game against the specified opponent. Return the game result (1 on win, 0 on draw and -1 on loss)."""
        if self.agent is None:
            raise ValueError("there needs to be an agent set for this manager")

        self.footsies_env.set_opponent(opponent.act if opponent is not None else None)

        env = self.env
        obs, info = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = self.agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
    
        return (float(reward) > 0) if not truncated else info["guard"][0] > info["guard"][1]

    def update_elo_sweep(self, episodes_per_opponent: int = 20):
        """Update Elos of all opponents and the agent."""
        regime = [(opponent, _) for opponent in self.opponent_history for _ in range(episodes_per_opponent)]
        for opponent, _ in tqdm(regime, desc="Elo sweep", unit="game", leave=False, dynamic_ncols=True, colour="#c651f0"):
            game_result = (self.play_game_against(opponent.opp) + 1) / 2
            self.update_elo(opponent, game_result)
        
    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()
