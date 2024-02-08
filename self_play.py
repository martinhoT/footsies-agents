from collections import deque
import random


class SelfPlayManager:
    def __init__(self,
        snapshot_method: callable,
        snapshot_frequency: int = 100,
        max_snapshots: int = 100,
        mix_bot: int = None,
    ):
        """
        Class that manages an opponent pool for self-play

        Parameters
        ----------
        snapshot_method: callable
            producer of agent snapshots to serve as future opponents
        self_play_snapshot_frequency: int
            how frequent to take a snapshot of the current policy for the opponent pool
        self_play_max_snapshots: int
            maximum capacity of the opponent pool. If at maximum, the oldest opponents are discarded
        self_play_mix_bot: int
            if specified, will include the in-game FOOTSIES bot as an opponent.
            Will enter after `self_play_mix_bot` episodes and stay for `self_play_mix_bot` episodes.
            As such, the opponent distribution will be 50/50, distributed between the snapshots and the in-game bot.
            This argument merely controls the switch frequency
        """
        
        self.snapshot_method = snapshot_method
        self.snapshot_frequency = snapshot_frequency
        self.max_snapshots = max_snapshots
        self.mix_bot = mix_bot

        self.opponent_pool = deque([], maxlen=max_snapshots)

        self.episode = 0
        self.mix_bot_counter = 0
        self.mix_bot_playing = False

        self._current_opponent = None

    def _add_opponent(self, opponent):
        self.opponent_pool.append(opponent)

    def _sample_opponent(self) -> "any":
        return random.sample(self.opponent_pool, 1)[0]

    def update_at_episode(self) -> bool:
        """Update the manager after every episode. Returns whether an opponent change was done"""
        self.episode += 1
        previous_opponent = self._current_opponent

        # Perform a snapshot of the agent at the current
        if self.episode % self.snapshot_frequency == 0:
            self._add_opponent(self.snapshot_method())
            print("Agent snapshot created!")

        self.mix_bot_counter += 1
        # Switch to the bot if the counter has surpassed the threshold
        if self.mix_bot is not None and self.mix_bot_counter >= self.mix_bot:
            # Go back to using opponent pool opponents
            if self.mix_bot_playing:
                self.mix_bot_counter = 0
                self.mix_bot_playing = False
                print("Will use opponents from the opponent pool now!")
        
            # Start using the in-game bot instead
            else:
                self._current_opponent = None
                self.mix_bot_counter = 0
                self.mix_bot_playing = True
                print("Will use the in-game opponent now!")
        
        # As long as the in-game bot is not playing, we will switch opponent every game
        if not self.mix_bot_playing:
            self._current_opponent = self._sample_opponent()

        return self._current_opponent != previous_opponent

    @property
    def current_opponent(self) -> callable:
        return self._current_opponent
