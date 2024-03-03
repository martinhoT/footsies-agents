import numpy as np
from collections import defaultdict


# TODO: double Q-learning support
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
    ):
        """
        Instantiate a Q-value table.

        Parameters
        ----------
        - `action_dim`: the number of possible actions
        - `opponent_action_dim`: the number of possible actions from the opponent.
        If not `None`, the opponent's immediate future actions `o` will be considered as part of the observation,
        and action-values will be of the form `Q(s, a, o)`
        - `learning_rate`: the learning rate
        - `discount`: the discount factor
        - `table_as_matrix`: whether the table should be stored as a matrix. If `False`, will be stored as a dictionary.
        Storing as a matrix should have a computational performance improvement, but spends much, much more memory than using a dictionary.
        A dictionary is recommended
        - `move_frame_n_bins`: how many separations to perform on the move frame observation variable when discretizing
        - `position_n_bins`: how many separations to perform on the position observation variable when discretizing
        """
        self.action_dim = action_dim
        self.opponent_action_dim = opponent_action_dim
        self.learning_rate = learning_rate
        self.discount = discount
        self.table_as_matrix = table_as_matrix

        self.obs_dim = 4**2 * 15**2 * move_frame_n_bins**2 * position_n_bins**2

        # Leave some leeway for the start and end of the linear space, since the start and end are part of the observation
        self.move_frame_bins = np.linspace(-0.1, 1.1, move_frame_n_bins)
        self.position_bins = np.linspace(-1.1, 1.1, position_n_bins)
        
        if self.table_as_matrix:
            if self.considering_opponent:
                self.table = np.zeros((self.obs_dim, self.opponent_action_dim, self.action_dim), dtype=np.float32)
                self.update_frequency_table = np.zeros((self.obs_dim, self.opponent_action_dim, self.action_dim), dtype=np.int32)
            else:
                self.table = np.zeros((self.obs_dim, self.action_dim), dtype=np.float32)
                self.update_frequency_table = np.zeros((self.obs_dim, self.action_dim), dtype=np.int32)
        else:
            if self.considering_opponent:
                self.table = defaultdict(lambda: np.zeros((self.opponent_action, self.action_dim), dtype=np.float32))
                self.update_frequency_table = defaultdict(lambda: np.zeros((self.opponent_action, self.action_dim), dtype=np.int32))
            else:
                self.table = defaultdict(lambda: np.zeros(self.action_dim, dtype=np.float32))
                self.update_frequency_table = defaultdict(lambda: np.zeros(self.action_dim, dtype=np.int32))

    def _obs_idx(self, obs: np.ndarray) -> int:
        gu1, gu2 = tuple(np.round(obs[0:2] * 3))
        mo1 = np.argmax(obs[2:17])
        mo2 = np.argmax(obs[17:32])
        mf1, mf2 = tuple(np.digitize(obs[32:34], self.move_frame_bins))
        po1, po2 = tuple(np.digitize(obs[34:36], self.position_bins))
        # yikes
        return int(gu1 + 4 * gu2 + 4 * 4 * mo1 + 4 * 4 * 15 * mo2 + 4 * 4 * 15 * 15 * mf1 + 4 * 4 * 15 * 15 * 5 * mf2 + 4 * 4 * 15 * 15 * 5 * 5 * po1 + 4 * 4 * 15 * 15 * 5 * 5 * 5 * po2)

    def update(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, terminated: bool, obs_opponent_action: int = None, next_obs_opponent_action: int = None) -> float:
        """
        Perform a Q-value update. Returns the TD error.
        
        If `opponent_action` is `None` and considering the opponent, then will perform an update for all opponent actions. This is useful for frame skipping.
        """
        if obs_opponent_action is None:
            obs_opponent_action = slice(None)
        if next_obs_opponent_action is None:
            next_obs_opponent_action = slice(None)

        obs = self._obs_idx(obs)
        next_obs = self._obs_idx(next_obs)

        nxt = (self.discount * np.max(self.qs(next_obs))) if not terminated else 0.0
        td_error = (reward + nxt - self.q(obs, action))

        if self.table_as_matrix:
            if self.considering_opponent:
                self.table[obs, obs_opponent_action, action] += self.learning_rate * td_error
                self.update_frequency_table[obs, obs_opponent_action, action] += 1
            else:
                self.table[obs, action] += self.learning_rate * td_error
                self.update_frequency_table[obs, action] += 1
        else:
            if self.considering_opponent:
                self.table[obs][obs_opponent_action, action] += self.learning_rate * td_error
                self.update_frequency_table[obs][obs_opponent_action, action] += 1
            else:
                self.table[obs][action] += self.learning_rate * td_error
                self.update_frequency_table[obs][action] += 1

        return td_error
    
    def q(self, obs: np.ndarray, action: int, opponent_action: int = None) -> float:
        """Get the Q-value for the given action and observation."""
        if opponent_action is None:
            opponent_action = slice(None)
        
        if self.table_as_matrix:
            return self.table[self._obs_idx(obs), opponent_action, action] if self.considering_opponent else self.table[self._obs_idx(obs), action]
        else:
            return self.table[self._obs_idx(obs)][opponent_action, action] if self.considering_opponent else self.table[self._obs_idx(obs)][action]

    def qs(self, obs: np.ndarray, opponent_action: int = None) -> np.ndarray:
        """Get the Q-values of all actions for the given observation."""
        if opponent_action is None:
            opponent_action = slice(None) 
        
        if self.table_as_matrix:
            return self.table[self._obs_idx(obs), opponent_action, :] if self.considering_opponent else self.table[self._obs_idx(obs), :]
        else:
            return self.table[self._obs_idx(obs)][opponent_action, :] if self.considering_opponent else self.table[self._obs_idx(obs)][:]

    def sample_action_best(self, obs: np.ndarray, opponent_action: int = None) -> int:
        """Sample the best action for the given observation."""
        if opponent_action is None and self.considering_opponent:
            raise ValueError("opponent_action must be provided when explicitly considering the opponent's actions")
        
        return np.argmax(self.qs(obs, opponent_action)) if self.considering_opponent else np.argmax(self.qs(obs))

    def sample_action_random(self, obs: np.ndarray, opponent_action: int = None) -> int:
        """Sample a random action for the given observation, with action probabilities proportional to their Q-values."""
        if opponent_action is None and self.considering_opponent:
            raise ValueError("opponent_action must be provided when explicitly considering the opponent's actions")
        
        qs = self.qs(obs, opponent_action) if self.considering_opponent else self.qs(obs)

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
