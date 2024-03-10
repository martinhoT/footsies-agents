import os
import numpy as np
import torch
import logging
from torch import nn
from abc import ABC, abstractmethod
from agents.torch_utils import create_layered_network

LOGGER = logging.getLogger("main.ql")


class QFunction(ABC):
    """Abstract class for a Q-value function estimator."""

    @abstractmethod
    def _update_q_value(self, obs: torch.Tensor, target: float, action: int = None, opponent_action: int = None):
        """Update the Q-value for the given state-action pair considering the provided TD error."""

    @abstractmethod
    def q(self, obs: torch.Tensor, action: int = None, opponent_action: int = None) -> float | torch.Tensor:
        """Get the Q-value for the given action and observation. If an action is `None`, then return Q-values considering all actions."""

    @abstractmethod
    def save(self, path: str):
        """Save the Q-function to a file stored in `path`."""
    
    @abstractmethod
    def load(self, path: str):
        """Load the Q-function from a file stored in `path`."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """The number of possible actions."""

    @property
    @abstractmethod
    def discount(self) -> float:
        """The discount factor."""

    @property
    @abstractmethod
    def considering_opponent(self) -> bool:
        """Whether we are explicitly considering the opponent's next action as part of the observation when computing a Q-value."""

    def update(
        self,
        obs: torch.Tensor,
        action: int,
        reward: float,
        next_obs: torch.Tensor,
        terminated: bool,
        obs_opponent_action: int = None,
        next_obs_opponent_action: int = None,
        next_obs_agent_action: int = None,
        policy: torch.Tensor = None,
        opponent_policy: torch.Tensor = None,
    ) -> float | torch.Tensor:
        """
        Perform a Q-value update. Returns the TD error.
        
        Special cases
        -------------
        - If `opponent_action` is `None` and considering the opponent, then will perform an update for all opponent actions. This is useful for frame skipping;
        - If `next_obs_agent_action` is not `None`, will consider the next Q-value for that action. This corresponds to SARSA;
        - If `policy` is not `None`, will consider the next Q-values weighted by the current policy. This corresponds to Expected SARSA;
        - If `opponent_policy` is not `None` and `next_obs_opponent_action` is `None`, importance sampling will be performed on the next Q-values considering how the opponent's policy differs from a uniform random opponent.

        In the absence of `next_obs_agent_action` and `policy`, this method performs a Q-learning update.
        """
        if next_obs_agent_action is not None and policy is not None:
            raise ValueError("'next_obs_agent_action' and 'policy' are mutually exclusive, as they dictate which kind of Q-value update to perform, but both were not None")
    
        next_q_values = self.q(next_obs, opponent_action=next_obs_opponent_action)
        if policy is not None:
            # We perform a weighted average of the next Q-values, corresponding to Expected SARSA.
            nxt = next_q_values @ policy
        elif next_obs_agent_action is not None:
            # We consider the next Q-value for the action that the agent took in the next step, corresponding to SARSA.
            nxt = next_q_values[:, next_obs_agent_action]
        else:
            # Performing the maximum over axis -1 is equivalent to performing the maximum over the last axis, which pertains to the axis of agent actions.
            # This corresponds to Q-learning.
            nxt = np.max(next_q_values, axis=-1, keepdims=True)
        
        
        if next_obs_opponent_action is None:
            # Importance sampling considering the opponent's behavior (policy).
            if opponent_policy is not None:
                nxt = self.discount * nxt @ (1 / (self.action_dim * opponent_policy))
            # This is a degenerate case. We will just take the average of the next Q-values, which is the same as giving a uniform random policy for the opponent policy.
            else:
                nxt = np.mean(nxt)
        else:
            # Importance sampling considering the opponent's behavior (policy).
            if opponent_policy is not None:
                nxt = self.action_dim * nxt[next_obs_opponent_action] / opponent_policy[next_obs_opponent_action]
            # If the opponent's behavior is not considered, then we will simply use the action the opponent took in the next step.
            else:
                nxt = nxt[next_obs_opponent_action]
            
        if opponent_policy is not None and next_obs_opponent_action is None:
            nxt = nxt
        nxt

        nxt_value = (self.discount * np.mean(nxt)) if not terminated else 0.0
        cur_value = self.q(obs, action, opponent_action=obs_opponent_action).clone().detach()
        target = reward + nxt_value
        td_error = (target - cur_value)

        # Don't even bother updating.
        if torch.all(td_error == 0.0):
            return td_error
        
        self._update_q_value(obs, target, action, obs_opponent_action)

        LOGGER.info(f"At an observation with agent action {action} and opponent action {obs_opponent_action} we had a Q-value of {cur_value}, now of {self.q(obs, action, opponent_action=obs_opponent_action)}, which was updated to {reward} + {nxt_value}")

        return td_error

    def sample_action_best(self, obs: torch.Tensor, opponent_action: int = None) -> int:
        """Sample the best action for the given observation."""
        if opponent_action is None and self.considering_opponent:
            raise ValueError("opponent_action must be provided when explicitly considering the opponent's actions")
        
        return np.argmax(self.q(obs, opponent_action=opponent_action)) if self.considering_opponent else np.argmax(self.q(obs))

    def sample_action_random(self, obs: torch.Tensor, opponent_action: int = None) -> int:
        """Sample a random action for the given observation, with action probabilities proportional to their Q-values."""
        if opponent_action is None and self.considering_opponent:
            raise ValueError("opponent_action must be provided when explicitly considering the opponent's actions")
        
        qs = self.q(obs, opponent_action=opponent_action) if self.considering_opponent else self.q(obs)

        # Softmax
        probs = torch.exp(qs) / torch.sum(torch.exp(qs))
        return np.random.choice(self.action_dim, p=probs)

    def sample_action_epsilon_greedy(self, obs: torch.Tensor, epsilon: float, opponent_action: int = None) -> int:
        """Sample an action following an epsilon-greedy policy."""
        if torch.rand() < epsilon:
            return torch.randint(self.action_size)
        else:
            return self.sample_action_best(obs, opponent_action)
    

# TODO: double Q-learning support
# TODO: forward view eligibility traces (maybe that's not possible with Q-learning? be careful)
# NOTE: Sutton & Barto chapter 6.8 talks about different ways of thinking about value and action-value functions. One example is afterstates. Maybe we are doing something along those lines by explicitly considering future actions of an opponent
#       We could use the afterstate thing if we had an environment model. That is actually possible with a deterministic environment such as fighting games. Evaluate this in the future.
class QFunctionTable(QFunction):
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

        self._action_dim = action_dim
        self.opponent_action_dim = opponent_action_dim
        self.learning_rate = learning_rate
        self._discount = discount
        self.table_as_matrix = table_as_matrix
        self.environment = environment

        if self.environment == "footsies":
            self.move_frame_n_bins = move_frame_n_bins
            self.position_n_bins = position_n_bins
            self._obs_dim = 4**2 * 15**2 * move_frame_n_bins**2 * position_n_bins**2

            # Leave some leeway for the start and end of the linear space, since the start and end are part of the observation
            self.move_frame_bins = np.linspace(-0.1, 1.1, move_frame_n_bins)
            self.position_bins = np.linspace(-1.1, 1.1, position_n_bins)

        elif self.environment == "mountain car":
            # These variables are hardcorded for the mountain car environment
            position_n_bins = 20
            velocity_n_bins = 20
            self.position_n_bins = position_n_bins
            self._obs_dim = position_n_bins * velocity_n_bins

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

    def _update_q_value(self, obs: np.ndarray, target: float, action: int = None, opponent_action: int = None):
        if action is None:
            action = slice(None)
        if opponent_action is None:
            opponent_action = slice(None)

        obs_idx = self._obs_idx(obs)

        td_error = (target - self.q(obs, action, opponent_action))

        if self.table_as_matrix:
            if self.considering_opponent:
                self.table[obs_idx, opponent_action, action] += self.learning_rate * td_error
                self.update_frequency_table[obs_idx, opponent_action, action] += 1
            else:
                self.table[obs_idx, action] += self.learning_rate * td_error
                self.update_frequency_table[obs_idx, action] += 1
        else:
            if self.considering_opponent:
                if obs_idx not in self.table:
                    self.table[obs_idx] = np.zeros((self.opponent_action_dim, self.action_dim), dtype=np.float32)
                    self.update_frequency_table[obs_idx] = np.zeros((self.opponent_action_dim, self.action_dim), dtype=np.int32)
                self.table[obs_idx][opponent_action, action] += self.learning_rate * td_error
                self.update_frequency_table[obs_idx][opponent_action, action] += 1
            else:
                if obs_idx not in self.table:
                    self.table[obs_idx] = np.zeros(self.action_dim, dtype=np.float32)
                    self.update_frequency_table[obs_idx] = np.zeros(self.action_dim, dtype=np.int32)
                self.table[obs_idx][action] += self.learning_rate * td_error
                self.update_frequency_table[obs_idx][action] += 1

    def q(self, obs: np.ndarray, action: int = None, opponent_action: int = None) -> float | np.ndarray:
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

    def update_frequency(self, obs: np.ndarray) -> np.ndarray:
        """Get the update frequency for the given observation."""
        if self.table_as_matrix:
            return self.update_frequency_table[self._obs_idx(obs), :, :] if self.considering_opponent else self.update_frequency_table[self._obs_idx(obs), :]
        else:
            obs_idx = self._obs_idx(obs)
            if obs_idx not in self.update_frequency_table:
                return np.zeros((self.opponent_action_dim, self.action_dim), dtype=np.int32) if self.considering_opponent else np.zeros(self.action_dim, dtype=np.int32)
            return self.update_frequency_table[obs_idx][:, :] if self.considering_opponent else self.update_frequency_table[obs_idx][:]

    def save(self, path: str):
        table_path = path + "_table"
        update_frequency_table_path = path + "_update_frequency_table"
        np.save(table_path, self.table)
        np.save(update_frequency_table_path, self.update_frequency_table)
    
    def load(self, path: str):
        table_path = path + "_table.npy"
        update_frequency_table_path = path + "_update_frequency_table.npy"
        self.table = np.load(table_path, allow_pickle=True).item()
        self.update_frequency_table = np.load(update_frequency_table_path, allow_pickle=True).item()

    @property
    def considering_opponent(self) -> bool:
        return self.opponent_action_dim is not None

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
        
    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def discount(self) -> float:
        return self._discount
        

class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: nn.Module = nn.Identity,
        representation: nn.Module = None,
    ):
        super().__init__()

        self._obs_dim = obs_dim
        self._action_dim = action_dim

        self.q_layers = create_layered_network(obs_dim, action_dim, hidden_layer_sizes, hidden_layer_activation)
        self.representation = nn.Identity() if representation is None else representation
    
    def forward(self, obs: torch.Tensor):
        rep = self.representation(obs)

        return self.q_layers(rep)

    def from_representation(self, rep: torch.Tensor) -> torch.Tensor:
        return self.q_layers(rep)

    @property
    def obs_dim(self) -> int:
        return self._obs_dim
    
    @property
    def action_dim(self) -> int:
        return self._action_dim


class QFunctionNetwork(QFunction):
    """Q-value table that uses a neural network to approximate the Q-values."""

    def __init__(
        self,
        q_network: QNetwork,
        action_dim: int,
        opponent_action_dim: int = None,
        discount: float = 1.0,
        learning_rate: float = 1e-2,
    ):
        """
        Instantiate a Q-value network.

        Parameters
        ----------
        - `q_network`: the neural network that will be used to approximate the Q-function
        - `action_dim`: the number of possible actions
        - `opponent_action_dim`: the number of possible actions from the opponent.
        If not `None`, the opponent's immediate future actions `o` will be considered as part of the observation,
        and action-values will be of the form `Q(s, o, a)`
        - `discount`: the discount factor
        - `learning_rate`: the learning rate
        """
        self._action_dim = action_dim
        self._opponent_action_dim = opponent_action_dim
        self._discount = discount

        self.q_network = q_network
        self.q_network_optimizer = torch.optim.SGD(self.q_network.parameters(), lr=learning_rate)

    def _append_opponent_action(self, obs: torch.Tensor, opponent_action: int) -> torch.Tensor:
        """Append the opponent's action to the observation."""
        opponent_action_oh = nn.functional.one_hot(torch.tensor([opponent_action]), num_classes=self._opponent_action_dim).float()
        return torch.hstack((obs, opponent_action_oh))

    def _update_q_value(self, obs: torch.Tensor, target: float, action: int = None, opponent_action: int = None):        
        obs = self._transform_obs(obs)

        if action is None:
            action = slice(None)

        self.q_network_optimizer.zero_grad()

        if opponent_action is None:
            loss = 0.0

            for o in range(self.action_dim):
                obs_with_opp = self._append_opponent_action(obs, o)
                predicted = self.q_network(obs_with_opp)
                loss += torch.mean((target - predicted[0, action])**2)
            loss.backward()
        
        else:
            obs_with_opp = self._append_opponent_action(obs, opponent_action)
            predicted = self.q_network(obs_with_opp)
            loss = torch.mean((target - predicted[0, action])**2)
            loss.backward()

        self.q_network_optimizer.step()
    
    def q(self, obs: torch.Tensor, action: int = None, opponent_action: int = None) -> float | torch.Tensor:
        if action is None:
            action = slice(None)

        if opponent_action is None:
            q_values = []
            for o in range(self.action_dim):
                obs_with_opp = self._append_opponent_action(obs, o)
                qs = self.q_network(obs_with_opp)[action]
                q_values.append(qs)

            q_values = np.vstack(q_values)

        else:
            obs_with_opp = self._append_opponent_action(obs, opponent_action)
            q_values = self.q_network(obs_with_opp)[action]

        return q_values

    def save(self, path: str):
        model_path = os.path.join(path, "qnetwork")
        torch.save(self.q_network.state_dict(), model_path)
    
    def load(self, path: str):
        model_path = os.path.join(path, "qnetwork")
        self.q_network.load_state_dict(torch.load(model_path))

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def discount(self) -> float:
        return self._discount

    @property
    def considering_opponent(self) -> bool:
        return self._opponent_action_dim is not None


class QFunctionNetworkDiscretized(QFunctionNetwork):
    """Q-value table that uses a neural network to approximate the Q-values. Observations are discretized in a similar way to the tabular Q-value estimation."""

    def __init__(
        self,
        q_network: QNetwork,
        action_dim: int,
        opponent_action_dim: int = None,
        discount: float = 1.0,
        learning_rate: float = 1e-2,
        move_frame_n_bins: int = 5,
        position_n_bins: int = 5,
        environment: str = "footsies",
        **kwargs,
    ):
        """
        Instantiate a Q-value network, using discretized observations.

        Parameters
        ----------
        - `q_network`: the neural network that will be used to approximate the Q-function.
        Should use the observation dimensionality appropriate for the environment, accessible through the relevant static method
        - `action_dim`: the number of possible actions
        - `opponent_action_dim`: the number of possible actions from the opponent.
        If not `None`, the opponent's immediate future actions `o` will be considered as part of the observation,
        and action-values will be of the form `Q(s, o, a)`
        - `discount`: the discount factor
        - `learning_rate`: the learning rate
        - `move_frame_n_bins`: how many separations to perform on the move frame observation variable when discretizing. Only valid for the "footsies" environment
        - `position_n_bins`: how many separations to perform on the position observation variable when discretizing. Only valid for the "footsies" environment
        - `environment`: the environment for which the Q-table is being instantiated. Currently only supports "footsies" and "mountain car"
        """
        super().__init__(
            q_network=q_network,
            action_dim=action_dim,
            opponent_action_dim=opponent_action_dim,
            discount=discount,
            learning_rate=learning_rate,
        )
        if environment not in ("footsies", "mountain car"):
            raise ValueError(f"environment '{environment}' not supported")

        if self.environment == "footsies":
            self.move_frame_n_bins = move_frame_n_bins
            self.position_n_bins = position_n_bins
            self._obs_dim = self.env_obs_dim("footsies", move_frame=move_frame_n_bins, position=position_n_bins)

            # Leave some leeway for the start and end of the linear space, since the start and end are part of the observation
            self.move_frame_bins = np.linspace(-0.1, 1.1, move_frame_n_bins)
            self.position_bins = np.linspace(-1.1, 1.1, position_n_bins)

        elif self.environment == "mountain car":
            # These variables are hardcorded for the mountain car environment
            position_n_bins = 20
            velocity_n_bins = 20
            self._obs_dim = self.env_obs_dim("mountain car", position=position_n_bins, velocity=velocity_n_bins)
            self.position_n_bins = position_n_bins

            self.position_bins = np.linspace(-1.2, 0.6, position_n_bins)
            self.velocity_bins = np.linspace(-0.07, 0.07, velocity_n_bins)

    def _transform_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Transform the continuous environment observation into a discretized one, similar in nature to the Q-table."""
        gu1, gu2 = tuple(torch.round(obs[0:2] * 3))
        mf1, mf2 = tuple(torch.digitize(obs[32:34], self.move_frame_bins))
        po1, po2 = tuple(torch.digitize(obs[34:36], self.position_bins))
        
        obs_discretized = torch.hstack((
            # Guard
            nn.functional.one_hot(torch.tensor([gu1, gu2]).long(), num_classes=4).float().flatten(),
            # Moves
            obs[2:32],
            # Move frame
            nn.functional.one_hot(torch.tensor([mf1, mf2]).long(), num_classes=self.move_frame_n_bins).float().flatten(),
            # Position
            nn.functional.one_hot(torch.tensor([po1, po2]).long(), num_classes=self.position_n_bins).float().flatten(),
        ))

        return obs_discretized.unsqueeze(0)

    def _update_q_value(self, obs_discretized: torch.Tensor, target: float, action: int = None, opponent_action: int = None):        
        obs_discretized = self._transform_obs(obs_discretized)
        super()._update_q_value(obs_discretized, target, action, opponent_action)
    
    def q(self, obs: torch.Tensor, action: int = None, opponent_action: int = None) -> float | torch.Tensor:
        obs_discretized = self._transform_obs(obs)
        return super().q(obs_discretized, action, opponent_action)

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @staticmethod
    def env_obs_dim(environment: str, **kwargs: dict) -> int:
        if environment == "footsies":
            move_frame = kwargs["move_frame"]
            position = kwargs["position"]
            return 4*2 + 15*2 + move_frame*2 + position*2
        elif environment == "mountain car":
            position = kwargs["position"]
            velocity = kwargs["velocity"]
            return position + velocity
