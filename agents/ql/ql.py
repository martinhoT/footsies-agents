import numpy as np


# TODO: double Q-learning support
# TODO: forward view eligibility traces (maybe that's not possible with Q-learning? be careful)
# NOTE: Sutton & Barto chapter 6.8 talks about different ways of thinking about value and action-value functions. One example is afterstates. Maybe we are doing something along those lines by explicitly considering future actions of an opponent
#       We could use the afterstate thing if we had an environment model. That is actually possible with a deterministic environment such as fighting games. Evaluate this in the future.
class QTable:
    """
    Implementation of a Q-value table.
    
    This implementation assumes that observations are flattened FOOTSIES observations, and won't work for any other environments.
    This is because the observation space is discretized into bins, and the way the discretization is performed is hard coded.
    """

    def __init__(
        self,
        action_dim: int,
        opponent_action_dim: int = None,
        discount: float = 1.0,
        learning_rate: float = 1e-2,
        table_as_matrix: bool = False,
        move_frame_n_bins: int = 5,
        position_n_bins: int = 5,
        environment: str = "footsies",
    ):
        """
        Instantiate a Q-value table.

        Parameters
        ----------
        - `action_dim`: the number of possible actions
        - `opponent_action_dim`: the number of possible actions from the opponent.
        If not `None`, the opponent's immediate future actions `o` will be considered as part of the observation,
        and action-values will be of the form `Q(s, o, a)`
        - `learning_rate`: the learning rate
        - `discount`: the discount factor
        - `table_as_matrix`: whether the table should be stored as a matrix. If `False`, will be stored as a dictionary.
        Storing as a matrix should have a computational performance improvement, but spends much, much more memory than using a dictionary.
        A dictionary is recommended
        - `move_frame_n_bins`: how many separations to perform on the move frame observation variable when discretizing. Only valid for the "footsies" environment
        - `position_n_bins`: how many separations to perform on the position observation variable when discretizing. Only valid for the "footsies" environment
        - `environment`: the environment for which the Q-table is being instantiated. Currently only supports "footsies" and "mountain car"
        """
        if environment not in ("footsies", "mountain car"):
            raise ValueError(f"environment '{environment}' not supported")

        self.action_dim = action_dim
        self.opponent_action_dim = opponent_action_dim
        self.learning_rate = learning_rate
        self.discount = discount
        self.table_as_matrix = table_as_matrix
        self.environment = environment

        if self.environment == "footsies":
            self.move_frame_n_bins = move_frame_n_bins
            self.position_n_bins = position_n_bins
            self.obs_dim = 4**2 * 15**2 * move_frame_n_bins**2 * position_n_bins**2

            # Leave some leeway for the start and end of the linear space, since the start and end are part of the observation
            self.move_frame_bins = np.linspace(-0.1, 1.1, move_frame_n_bins)
            self.position_bins = np.linspace(-1.1, 1.1, position_n_bins)

        elif self.environment == "mountain car":
            # These variables are hardcorded for the mountain car environment
            position_n_bins = 20
            velocity_n_bins = 20
            self.position_n_bins = position_n_bins
            self.obs_dim = position_n_bins * velocity_n_bins

            self.position_bins = np.linspace(-1.2, 0.6, position_n_bins)
            self.velocity_bins = np.linspace(-0.07, 0.07, velocity_n_bins)

        if self.table_as_matrix:
            if self.considering_opponent:
                self.table = np.zeros((self.obs_dim, self.opponent_action_dim, self.action_dim), dtype=np.float32)
                self.update_frequency_table = np.zeros((self.obs_dim, self.opponent_action_dim, self.action_dim), dtype=np.int32)
            else:
                self.table = np.zeros((self.obs_dim, self.action_dim), dtype=np.float32)
                self.update_frequency_table = np.zeros((self.obs_dim, self.action_dim), dtype=np.int32)
        else:
            if self.considering_opponent:
                self.table = {}
                self.update_frequency_table = {}
                self.table_empty_value = np.zeros((self.opponent_action_dim, self.action_dim), dtype=np.float32)
            else:
                self.table = {}
                self.update_frequency_table = {}
                self.table_empty_value = np.zeros(self.action_dim, dtype=np.float32)

            # Make the empty value read-only. This is the value that is returned if the queried observation is not in the Q-table
            self.table_empty_value.setflags(write=False)
        
    def _footsies_obs_idx(self, obs: np.ndarray) -> int:
        """Obtain the integer identifier associated to the given observation from the `footsies` environment."""
        gu1, gu2 = tuple(np.round(obs[0:2] * 3))
        mo1 = np.argmax(obs[2:17])
        mo2 = np.argmax(obs[17:32])
        mf1, mf2 = tuple(np.digitize(obs[32:34], self.move_frame_bins))
        po1, po2 = tuple(np.digitize(obs[34:36], self.position_bins))
        # yikes
        return int(
            gu1
            + 4 * gu2
            + 4**2 * mo1
            + 4**2 * 15 * mo2
            + 4**2 * 15**2 * mf1
            + 4**2 * 15**2 * self.move_frame_n_bins * mf2
            + 4**2 * 15**2 * self.move_frame_n_bins**2 * po1
            + 4**2 * 15**2 * self.move_frame_n_bins**2 * self.position_n_bins * po2
        )

    def _mountain_car_obs_idx(self, obs: np.ndarray) -> int:
        """Obtain the integer identifier associated to the given observation from the `mountain car` environment."""
        pos = np.digitize(obs[0], self.position_bins).item()
        vel = np.digitize(obs[1], self.velocity_bins).item()
        return int(pos + 10 * vel)

    def _obs_idx(self, obs: np.ndarray) -> int:
        """Obtain the integer identifier associated to the given observation."""
        if self.environment == "footsies":
            return self._footsies_obs_idx(obs)
        elif self.environment == "mountain car":
            return self._mountain_car_obs_idx(obs)
        
        return None
        
    def update(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, terminated: bool, obs_opponent_action: int = None, next_obs_opponent_action: int = None) -> float | np.ndarray:
        """
        Perform a Q-value update. Returns the TD error.
        
        If `opponent_action` is `None` and considering the opponent, then will perform an update for all opponent actions. This is useful for frame skipping.
        """
        if action is None:
            action = slice(None)
        if obs_opponent_action is None:
            obs_opponent_action = slice(None)
        if next_obs_opponent_action is None:
            next_obs_opponent_action = slice(None)

        # Performing the maximum over axis -1 is equivalent to performing the maximum over the last axis, which pertains to the axis of agent actions.
        nxt = np.max(self.q(next_obs, opponent_action=next_obs_opponent_action), axis=-1, keepdims=True)
        # We expect the Q-value of any next opponent action to be the same in case one in particular wasn't specified.
        # If that's not the case, then updates must not have been done correctly.
        # if not np.all(nxt == nxt[0]):
            # raise RuntimeError("the Q-value of the next state should be the same for all next opponent actions, in case one in particular is not being considered")
        # However, since small variations in the move progress determine whether a move is actionable or not, but here we are binning them into the same slot, we can't really have these expectations.
        # Therefore, we will just take the average.
        nxt_value = (self.discount * np.mean(nxt)) if not terminated else 0.0
        td_error = (reward + nxt_value - self.q(obs, action, opponent_action=obs_opponent_action))
        
        # Don't even bother updating. This also saves needlessly creating matrices with 0s when using a dictionary
        if np.all(td_error == 0.0):
            return td_error

        obs_idx = self._obs_idx(obs)

        if self.table_as_matrix:
            if self.considering_opponent:
                self.table[obs_idx, obs_opponent_action, action] += self.learning_rate * td_error
                self.update_frequency_table[obs_idx, obs_opponent_action, action] += 1
            else:
                self.table[obs_idx, action] += self.learning_rate * td_error
                self.update_frequency_table[obs_idx, action] += 1
        else:
            if self.considering_opponent:
                if obs_idx not in self.table:
                    self.table[obs_idx] = np.zeros((self.opponent_action_dim, self.action_dim), dtype=np.float32)
                    self.update_frequency_table[obs_idx] = np.zeros((self.opponent_action_dim, self.action_dim), dtype=np.int32)
                self.table[obs_idx][obs_opponent_action, action] += self.learning_rate * td_error
                self.update_frequency_table[obs_idx][obs_opponent_action, action] += 1
            else:
                if obs_idx not in self.table:
                    self.table[obs_idx] = np.zeros(self.action_dim, dtype=np.float32)
                    self.update_frequency_table[obs_idx] = np.zeros(self.action_dim, dtype=np.int32)
                self.table[obs_idx][action] += self.learning_rate * td_error
                self.update_frequency_table[obs_idx][action] += 1

        return td_error
    
    def q(self, obs: np.ndarray, action: int = None, opponent_action: int = None) -> float | np.ndarray:
        """Get the Q-value for the given action and observation. If an action is `None`, then return Q-values considering all actions"""
        if action is None:
            action = slice(None)
        if opponent_action is None:
            opponent_action = slice(None)
        
        if self.table_as_matrix:
            return self.table[self._obs_idx(obs), opponent_action, action] if self.considering_opponent else self.table[self._obs_idx(obs), action]
        else:
            obs_idx = self._obs_idx(obs)
            if obs_idx not in self.table:
                return self.table_empty_value[opponent_action, action]
            return self.table[self._obs_idx(obs)][opponent_action, action] if self.considering_opponent else self.table[self._obs_idx(obs)][action]

    def sample_action_best(self, obs: np.ndarray, opponent_action: int = None) -> int:
        """Sample the best action for the given observation."""
        if opponent_action is None and self.considering_opponent:
            raise ValueError("opponent_action must be provided when explicitly considering the opponent's actions")
        
        return np.argmax(self.q(obs, opponent_action=opponent_action)) if self.considering_opponent else np.argmax(self.q(obs))

    def sample_action_random(self, obs: np.ndarray, opponent_action: int = None) -> int:
        """Sample a random action for the given observation, with action probabilities proportional to their Q-values."""
        if opponent_action is None and self.considering_opponent:
            raise ValueError("opponent_action must be provided when explicitly considering the opponent's actions")
        
        qs = self.q(obs, opponent_action=opponent_action) if self.considering_opponent else self.q(obs)

        # Softmax
        probs = np.exp(qs) / np.sum(np.exp(qs)) 
        return np.random.choice(self.action_dim, p=probs)

    def sample_action_epsilon_greedy(self, obs: np.ndarray, epsilon: float, opponent_action: int = None) -> int:
        """Sample an action following an epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        else:
            return self.sample_action_best(obs, opponent_action)
    
    @property
    def considering_opponent(self) -> bool:
        """Whether we are explicitly considering the opponent's next action as part of the observation when computing a Q-value."""
        return self.opponent_action_dim is not None

    def save(self, path: str):
        """Save the Q-table to a file stored in `path`."""
        table_path = path + "_table"
        update_frequency_table_path = path + "_update_frequency_table"
        np.save(table_path, self.table)
        np.save(update_frequency_table_path, self.update_frequency_table)
    
    def load(self, path: str):
        """Load the Q-table from a file stored in `path`."""
        table_path = path + "_table.npy"
        update_frequency_table_path = path + "_update_frequency_table.npy"
        self.table = np.load(table_path, allow_pickle=True).item()
        self.update_frequency_table = np.load(update_frequency_table_path, allow_pickle=True).item()

    def sparsity(self) -> float:
        """Calculate how sparse the Q-table is (i.e. how many entries are 0)."""
        if self.table_as_matrix:
            if self.table.size == 0:
                return 0.0
            return 1 - len(self.table.nonzero()[0]) / self.table.size
        else:
            if len(self.table) == 0:
                return 0.0
            return sum(np.sum(m == 0.0) for m in self.table.values()) / (len(self.table) * self.action_dim**2)