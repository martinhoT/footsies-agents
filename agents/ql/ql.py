import os
import numpy as np
import torch
import logging
import copy
from torch import nn
from abc import ABC, abstractmethod
from agents.torch_utils import create_layered_network
from agents.torch_utils import ToMatrix

LOGGER = logging.getLogger("main.ql")


class QFunction(ABC):
    """Abstract class for a Q-value function estimator."""

    @abstractmethod
    def _update_q_value(self, obs: torch.Tensor, target: float, action: int = None, opponent_action: int = None):
        """Update the Q-value for the given state-action pair considering the provided TD error."""

    @abstractmethod
    def q(self, obs: torch.Tensor, action: int = None, opponent_action: int = None) -> torch.Tensor:
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
    def opponent_action_dim(self) -> int:
        """The number of possible actions from the opponent."""

    @property
    @abstractmethod
    def discount(self) -> float:
        """The discount factor."""

    @discount.setter
    @abstractmethod
    def discount(self):
        """Change the discount factor."""

    @property
    @abstractmethod
    def considering_opponent(self) -> bool:
        """Whether we are explicitly considering the opponent's next action as part of the observation when computing a Q-value."""

    def update(
        self,
        obs: torch.Tensor,
        reward: float,
        next_obs: torch.Tensor,
        terminated: bool,
        agent_action: int = None,
        opponent_action: int = None,
        next_agent_action: int = None,
        next_opponent_action: int = None,
        next_agent_policy: torch.Tensor = None,
        next_opponent_policy: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Perform a Q-value update. Returns the TD error.
        
        Parameters
        ----------
        - `obs`: the current observation
        - `reward`: the reward obtained after taking the action `agent_action` and the opponent taking the action `opponent_action`
        - `next_obs`: the next observation
        - `terminated`: whether the episode has terminated normally (i.e. `next_obs` is a terminal observation)
        - `agent_action`: the action taken by the agent. If `None`, will perform frameskipping, considering all actions
        - `opponent_action`: the action taken by the opponent. If `None`, will perform frameskipping, considering all actions
        - `next_agent_action`: the action taken by the agent in the next step. If not `None`, will consider the next Q-value for that action. This corresponds to SARSA
        - `next_opponent_action`: the action taken by the opponent in the next step. If not `None`, will consider the next Q-value for that opponent action.
        WARNING: probably requires importance sampling, so it should not be used
        - `next_agent_policy`: the agent's policy evaluated at the next observation. If not `None`, will consider the next Q-values weighted by the current policy.
        This corresponds to Expected SARSA, or Q-learning if `agent_policy` happens to be greedy.
        Should be a matrix with agent actions in the rows and opponent actions in the columns
        - `next_opponent_policy`: the opponent's policy evaluated at the next observation. Should be a column vector
        If not `None`, will weight the next Q-values according to how likely they are to happen under the opponent's policy.
        If both this value and `next_opponent_action` are `None`, then the opponent will be assumed to follow a unifrom random policy in the next observation
        """
        if terminated:
            nxt_value = 0.0
        
        else:
            if next_agent_action is not None and next_agent_policy is not None:
                raise ValueError("'next_agent_action' and 'next_agent_policy' are mutually exclusive, as they dictate which kind of Q-value update to perform, but both were not None")
            
            if next_opponent_action is not None and next_opponent_policy is not None:
                raise ValueError("'next_opponent_action' and 'next_opponent_policy' are mutually exclusive, as they dictate which kind of Q-value update to perform, but both were not None")
            
            # Lazy formulation to save lines of code
            if next_opponent_action is not None:
                next_opponent_policy = nn.functional.one_hot(torch.tensor([next_opponent_action]), num_classes=self.action_dim).unsqueeze(1).float()
                LOGGER.debug("Will consider a single action from the opponent in the update")
            elif next_opponent_policy is None:
                next_opponent_policy = torch.ones(self.action_dim).unsqueeze(1) / self.action_dim
                LOGGER.debug("Will consider the opponent to be following a uniform random policy in the update")
            else:
                LOGGER.debug("Will consider a custom opponent's policy in the update")
            if next_agent_action is not None:
                next_agent_policy = nn.functional.one_hot(torch.tensor([next_agent_action]), num_classes=self.action_dim).unsqueeze(1).float()
                LOGGER.debug("Will perform a SARSA update")

            # The target doesn't need gradients to flow through. We also squeeze, assuming only one observation is being considered in next_obs
            next_qs = self.q(next_obs).detach().squeeze(dim=0)
            # Perform SARSA-like update
            if next_agent_policy is not None:
                next_q = next_opponent_policy.T @ torch.diag(next_qs @ next_agent_policy).unsqueeze(1)
                LOGGER.debug("Setup SARSA-like update")
            # Perform Q-learning update
            else:
                next_q = next_opponent_policy @ np.max(next_qs, axis=-1, keepdims=True)
                LOGGER.debug("Setup Q-learning update")

            nxt_value = self.discount * next_q

        target = reward + nxt_value
        cur_value = self.q(obs, agent_action, opponent_action=opponent_action).detach()
        td_error = target - cur_value

        if torch.any(torch.isnan(td_error)):
            LOGGER.critical("Q-values are NaN! Training probably got unstable due to improper targets or high learning rate")

        # Don't even bother updating.
        if torch.all(td_error == 0.0):
            return td_error
        
        self._update_q_value(obs, target, agent_action, opponent_action)

        if LOGGER.isEnabledFor(logging.DEBUG):
            new_value = self.q(obs, agent_action, opponent_action=opponent_action)
            LOGGER.debug(f"At an observation with agent action %s and opponent action %s we had a Q-value of %s, now of %s, which was updated to %s + %s",
                agent_action, opponent_action, cur_value, new_value, reward, nxt_value
            )

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
        self._opponent_action_dim = opponent_action_dim
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
                self.table = np.zeros((self.obs_dim, self._opponent_action_dim, self.action_dim), dtype=np.float32)
                self.update_frequency_table = np.zeros((self.obs_dim, self._opponent_action_dim, self.action_dim), dtype=np.int32)
            else:
                self.table = np.zeros((self.obs_dim, self.action_dim), dtype=np.float32)
                self.update_frequency_table = np.zeros((self.obs_dim, self.action_dim), dtype=np.int32)
        else:
            if self.considering_opponent:
                self.table = {}
                self.update_frequency_table = {}
                self.table_empty_value = np.zeros((self._opponent_action_dim, self.action_dim), dtype=np.float32)
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
                    self.table[obs_idx] = np.zeros((self._opponent_action_dim, self.action_dim), dtype=np.float32)
                    self.update_frequency_table[obs_idx] = np.zeros((self._opponent_action_dim, self.action_dim), dtype=np.int32)
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
                return np.zeros((self._opponent_action_dim, self.action_dim), dtype=np.int32) if self.considering_opponent else np.zeros(self.action_dim, dtype=np.int32)
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
        return self._opponent_action_dim is not None

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
    def opponent_action_dim(self) -> int:
        return self._opponent_action_dim

    @property
    def discount(self) -> float:
        return self._discount
    
    @discount.setter
    def discount(self, discount: float):
        self._discount = discount
        

class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: nn.Module = nn.Identity,
        representation: nn.Module = None,
        is_footsies: bool = False,
        use_dense_reward: bool = True, # Dictates the range of possible output values
        opponent_action_dim: int = None,
    ):
        super().__init__()

        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._opponent_action_dim = opponent_action_dim

        self._output_multiplier = 2.0 if use_dense_reward else 1.0

        output_dim = action_dim * (opponent_action_dim if opponent_action_dim is not None else 0)
        self.q_layers = create_layered_network(obs_dim, output_dim, hidden_layer_sizes, hidden_layer_activation)
        if opponent_action_dim is not None:
            self.q_layers.append(ToMatrix(opponent_action_dim, action_dim))
        if is_footsies:
            self.q_layers.append(nn.Tanh())
        self.representation = nn.Identity() if representation is None else representation
    
    def forward(self, obs: torch.Tensor):
        rep = self.representation(obs)
        qs = self.q_layers(rep)
        return qs * self._output_multiplier

    def from_representation(self, rep: torch.Tensor) -> torch.Tensor:
        qs = self.q_layers(rep)
        return qs * self._output_multiplier

    @property
    def obs_dim(self) -> int:
        return self._obs_dim
    
    @property
    def action_dim(self) -> int:
        return self._action_dim
    
    @property
    def opponent_action_dim(self) -> int:
        return self._opponent_action_dim


class QFunctionNetwork(QFunction):
    """Q-value table that uses a neural network to approximate the Q-values."""

    def __init__(
        self,
        q_network: QNetwork,
        action_dim: int,
        opponent_action_dim: int = None,
        discount: float = 1.0,
        learning_rate: float = 1e-2,
        target_network: QNetwork = None,
        target_network_blank: bool = False,
        target_network_update_interval: int = 1000,
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
        - `target_network`: an extra, slower-changing, target network to stabilize learning. If `None`, it won't be used
        - `target_network_blank`: if `True`, the target network will be a copy the Q-network, but with all parameters set to 0, otherwise it's just a copy
        - `target_network_update_interval`: the interval between target network updates, in terms of update steps
        """
        self._action_dim = action_dim
        self._opponent_action_dim = opponent_action_dim
        self._discount = discount
        self._use_target_network = target_network is not None
        self._target_network_update_interval = target_network_update_interval

        self.q_network = q_network
        self.q_network_optimizer = torch.optim.SGD(self.q_network.parameters(), lr=learning_rate)
        if self._use_target_network:
            # Make a copy of the Q-network, but with parameters set to 0 to remove any noise that the Q-network would pick up
            self.target_network = target_network
            if target_network_blank:
                for param in self.target_network.parameters():
                    param.data.zero_()
            else:
                for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                    target_param.data.copy_(q_param.data)
            
            # Disable gradient computation to get speed boost.
            self.target_network.requires_grad_(False)
        else:
            self.target_network = self.q_network
        
        # Target network update tracker
        self._current_update_step = 0

    def _update_q_value(self, obs: torch.Tensor, target: float, action: int = None, opponent_action: int = None):        
        if action is None:
            action = slice(None)
        if opponent_action is None:
            opponent_action = slice(None)

        self.q_network_optimizer.zero_grad()

        predicted = self.q_network(obs)
        loss = torch.mean((target - predicted[:, opponent_action, action])**2)
        loss.backward()

        self.q_network_optimizer.step()

        self._current_update_step += 1
        if self._use_target_network and self._current_update_step >= self._target_network_update_interval:
            for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(q_param.data)
            self._current_update_step = 0
    
    def q(self, obs: torch.Tensor, action: int = None, opponent_action: int = None) -> float | torch.Tensor:
        if action is None:
            action = slice(None)
        if opponent_action is None:
            opponent_action = slice(None)

        return self.target_network(obs)[:, opponent_action, action]

    def save(self, path: str):
        model_path = os.path.join(path, "qnetwork")
        torch.save(self.q_network.state_dict(), model_path)
    
    def load(self, path: str):
        model_path = os.path.join(path, "qnetwork")
        state_dict = torch.load(model_path)
        self.q_network.load_state_dict(state_dict)
        if self._use_target_network:
            self.target_network.load_state_dict(state_dict)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def opponent_action_dim(self) -> int:
        return self._opponent_action_dim

    @property
    def discount(self) -> float:
        return self._discount

    @discount.setter
    def discount(self, discount: float):
        self._discount = discount

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
