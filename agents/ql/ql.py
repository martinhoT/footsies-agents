import os
import numpy as np
import torch as T
import torch.nn.functional as F
import logging
from torch import nn
from typing import Literal
from abc import ABC, abstractmethod
from agents.torch_utils import create_layered_network
from agents.torch_utils import ToMatrix
from footsies_gym.moves import FootsiesMove
from agents.action import ActionMap

LOGGER = logging.getLogger("main.ql")


class QFunction(ABC):
    """Abstract class for a Q-value function estimator."""

    @abstractmethod
    def _update_q_value(self, obs: T.Tensor, target: float, action: int | None = None, opponent_action: int | None = None):
        """Update the Q-value for the given state-action pair considering the provided TD error."""

    @abstractmethod
    def q(self, obs: T.Tensor, action: int | None = None, opponent_action: int | None = None) -> T.Tensor:
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
    def discount(self, value: float):
        """"""

    @property
    @abstractmethod
    def learning_rate(self) -> float:
        """The learning rate."""

    @learning_rate.setter
    @abstractmethod
    def learning_rate(self, value: float):
        """"""

    @property
    @abstractmethod
    def considering_opponent(self) -> bool:
        """Whether we are explicitly considering the opponent's next action as part of the observation when computing a Q-value."""

    def update(
        self,
        obs: T.Tensor,
        reward: float,
        next_obs: T.Tensor,
        terminated: bool,
        agent_action: int | None = None,
        opponent_action: int | None = None,
        next_agent_policy: T.Tensor | Literal["uniform", "greedy"] = "greedy",
        next_opponent_policy: T.Tensor | Literal["uniform", "greedy"] = "greedy",
    ) -> T.Tensor:
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
        WARNING: probably requires importance sampling, so it should not be used
        - `next_agent_policy`: the agent's policy evaluated at the next observation.
        Should be a matrix with agent actions in the rows and opponent actions in the columns.
        This corresponds to Expected SARSA, or Q-learning if `agent_policy` happens to be greedy.
        Can also be a string: "greedy" for a greedy policy, or "uniform" for a uniform random policy.
        - `next_opponent_policy`: the opponent's policy evaluated at the next observation.
        Should be a column vector.
        If both this value and `next_opponent_action` are `None`, then the opponent will be assumed to follow a uniform random policy in the next observation.
        Can also be a string: "greedy" for a greedy policy, or "uniform" for a uniform random policy.
        """
        if isinstance(next_opponent_policy, T.Tensor) and next_opponent_policy.dim() > 2:
            raise ValueError("expected next_opponent_policy to be a column vector, but found a >2D tensor instead")
        if self.opponent_action_dim is None:
            raise ValueError("not considering opponent actions *strongly* is not supported")

        if terminated:
            nxt_value = 0.0
        
        else:
            # The target doesn't need gradients to flow through. We also squeeze, assuming only one observation is being considered in next_obs.
            # If using a neural network, then we may need to consider using a target network to avoid instability.
            next_qs = self.q(next_obs, use_target_network=True) if isinstance(self, QFunctionNetwork) else self.q(next_obs)
            next_qs = next_qs.detach().squeeze(dim=0)

            # First compute the effective agent policy (a matrix or column vector), if needed.
            # If it was explicitly passed, then we don't need to artificially create one.
            if next_agent_policy == "uniform":
                next_agent_policy = T.ones(self.action_dim).unsqueeze(1) / self.action_dim
            elif next_agent_policy == "greedy":
                next_agent_policy = F.one_hot(T.argmax(next_qs, dim=-1), num_classes=self.action_dim).float().T

            # The next Q-values aggregated according to the agent's policy, but missing the opponent's
            next_q_opp = T.sum(next_qs * next_agent_policy.T, dim=1, keepdim=True)

            # Then compute the effective opponent policy (a column vector), if needed.
            # If it was explicitly passed, then we don't need to artificially create one.
            if next_opponent_policy == "uniform":
                next_opponent_policy = T.ones(self.opponent_action_dim).unsqueeze(1) / self.opponent_action_dim
            elif next_opponent_policy == "greedy":
                # This one is tricky! The opponent's policy is a column vector (not conditioned on agent action),
                # so we need to implicitly take the agent's policy into account in the Q-values
                next_opponent_policy = F.one_hot(next_q_opp.argmin(), num_classes=self.opponent_action_dim).unsqueeze(1).float()

            # Finally, aggregate the next Q-values according to the opponent's policy
            next_q = next_opponent_policy.T @ next_q_opp
            
            nxt_value = self.discount * next_q.item()

        target = reward + nxt_value
        cur_value = self.q(obs, agent_action, opponent_action=opponent_action).detach()
        td_error = target - cur_value

        if T.any(T.isnan(td_error)):
            LOGGER.critical("Q-values are NaN! Training probably got unstable due to improper targets or high learning rate")

        # Don't even bother updating.
        if T.all(td_error == 0.0):
            return td_error
        
        self._update_q_value(obs, target, agent_action, opponent_action)

        if LOGGER.isEnabledFor(logging.DEBUG):
            # We don't use the target network, in case it's the neural network implementation, in order to check whether anything was updated at all
            if isinstance(self, QFunctionNetwork):
                new_value = self.q(obs, agent_action, opponent_action=opponent_action, use_target_network=False)
            else:
                new_value = self.q(obs, agent_action, opponent_action=opponent_action)
            LOGGER.debug(f"At an observation with agent action %s and opponent action %s we had a Q-value of %s, now of %s, which was updated to %s + %s",
                agent_action, opponent_action, cur_value, new_value, reward, nxt_value
            )

        return td_error

    def sample_action_best(self, obs: T.Tensor, opponent_action: int | None = None) -> int:
        """Sample the best action for the given observation."""
        if opponent_action is None and self.considering_opponent:
            raise ValueError("opponent_action must be provided when explicitly considering the opponent's actions")
        
        qs = self.q(obs, opponent_action=opponent_action) if self.considering_opponent else self.q(obs)
        return int(qs.argmax().item())

    def sample_action_random(self, obs: T.Tensor, opponent_action: int | None = None) -> int:
        """Sample a random action for the given observation, with action probabilities proportional to their Q-values."""
        if opponent_action is None and self.considering_opponent:
            raise ValueError("opponent_action must be provided when explicitly considering the opponent's actions")
        
        qs = self.q(obs, opponent_action=opponent_action) if self.considering_opponent else self.q(obs)

        # Softmax
        probs = T.exp(qs) / T.sum(T.exp(qs))
        return np.random.choice(self.action_dim, p=probs)

    def sample_action_epsilon_greedy(self, obs: T.Tensor, epsilon: float, opponent_action: int | None = None) -> int:
        """Sample an action following an epsilon-greedy policy."""
        if T.rand() < epsilon:
            return int(T.randint(0, self.action_dim, (1,)).item())
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
        opponent_action_dim: int | None = None,
        discount: float = 1.0,
        learning_rate: float = 1e-2,
        table_as_matrix: bool = False,
        position_n_bins: int = 10,
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
        - `position_n_bins`: how many separations to perform on the position observation variable when discretizing. Only valid for the "footsies" environment
        - `environment`: the environment for which the Q-table is being instantiated. Currently only supports "footsies" and "mountain car"
        """
        if environment not in ("footsies", "mountain car"):
            raise ValueError(f"environment '{environment}' not supported")

        self._action_dim = action_dim
        self._opponent_action_dim = opponent_action_dim
        self._learning_rate = learning_rate
        self._discount = discount
        self.environment = environment

        if self.environment == "footsies":
            self.position_n_bins = position_n_bins
            self.max_move_duration = max(move.value.duration for move in FootsiesMove)
            self._obs_dim = 4**2 * 15**2 * self.max_move_duration**2 * position_n_bins**2

            # Leave some leeway for the start and end of the linear space, since the start and end are part of the observation
            self.position_bins = np.linspace(-1.1, 1.1, position_n_bins)

        elif self.environment == "mountain car":
            # These variables are hardcorded for the mountain car environment
            position_n_bins = 20
            velocity_n_bins = 20
            self.position_n_bins = position_n_bins
            self._obs_dim = position_n_bins * velocity_n_bins

            self.position_bins = np.linspace(-1.2, 0.6, position_n_bins)
            self.velocity_bins = np.linspace(-0.07, 0.07, velocity_n_bins)

        self.table: T.Tensor | dict[int | None, T.Tensor]
        self.update_frequency_table: T.Tensor | dict[int | None, T.Tensor]
        if table_as_matrix:
            if self._opponent_action_dim is not None:
                self.table = T.zeros((self.obs_dim, self._opponent_action_dim, self.action_dim), dtype=T.float32)
                self.update_frequency_table = T.zeros((self.obs_dim, self._opponent_action_dim, self.action_dim), dtype=T.int32)
            else:
                self.table = T.zeros((self.obs_dim, self.action_dim), dtype=T.float32)
                self.update_frequency_table = T.zeros((self.obs_dim, self.action_dim), dtype=T.int32)
        else:
            if self._opponent_action_dim is not None:
                self.table = {}
                self.update_frequency_table = {}
                self.table_empty_value = T.zeros((self._opponent_action_dim, self.action_dim), dtype=T.float32)
            else:
                self.table = {}
                self.update_frequency_table = {}
                self.table_empty_value = T.zeros(self.action_dim, dtype=T.float32)
    
    def _footsies_obs_idx(self, obs: T.Tensor) -> int:
        """Obtain the integer identifier associated to the given observation from the `footsies` environment."""
        obs = obs.squeeze().numpy()
        
        gu1, gu2 = tuple(np.round(obs[0:2] * 3))
        mo1 = np.argmax(obs[2:17])
        mo2 = np.argmax(obs[17:32])

        mo1_move = ActionMap.move_from_move_index(mo1.item())
        mo2_move = ActionMap.move_from_move_index(mo2.item())

        mf1 = round(obs[32].item() * mo1_move.value.duration)
        mf2 = round(obs[33].item() * mo2_move.value.duration)

        po1, po2 = tuple(np.digitize(obs[34:36], self.position_bins))
        
        # yikes
        return int(
            gu1
            + 4 * gu2
            + 4**2 * mo1
            + 4**2 * 15 * mo2
            + 4**2 * 15**2 * mf1
            + 4**2 * 15**2 * self.max_move_duration * mf2
            + 4**2 * 15**2 * self.max_move_duration**2 * po1
            + 4**2 * 15**2 * self.max_move_duration**2 * self.position_n_bins * po2
        )

    def _mountain_car_obs_idx(self, obs: T.Tensor) -> int:
        """Obtain the integer identifier associated to the given observation from the `mountain car` environment."""
        obs = obs.squeeze().numpy()
        
        pos = np.digitize(obs[0], self.position_bins).item()
        vel = np.digitize(obs[1], self.velocity_bins).item()
        return int(pos + 10 * vel)

    def _obs_idx(self, obs: T.Tensor) -> int | None:
        """Obtain the integer identifier associated to the given observation."""
        if self.environment == "footsies":
            return self._footsies_obs_idx(obs)
        elif self.environment == "mountain car":
            return self._mountain_car_obs_idx(obs)
        
        return None

    def _update_q_value(self, obs: T.Tensor, target: float, action: int | None = None, opponent_action: int | None = None):
        action_idx = slice(None) if action is None else action
        opponent_action_idx = slice(None) if opponent_action is None else opponent_action

        obs_idx = self._obs_idx(obs)

        td_error = (target - self.q(obs, action, opponent_action).squeeze(dim=0).numpy(force=True))

        LOGGER.debug("Update Q-value of observation %s to %s", obs_idx, target)

        if isinstance(self.table, T.Tensor) and isinstance(self.update_frequency_table, T.Tensor):
            if self._opponent_action_dim is not None:
                self.table[obs_idx, opponent_action_idx, action_idx] += self._learning_rate * td_error
                self.update_frequency_table[obs_idx, opponent_action_idx, action_idx] += 1
            else:
                self.table[obs_idx, action_idx] += self._learning_rate * td_error
                self.update_frequency_table[obs_idx, action_idx] += 1
        else:
            if self._opponent_action_dim is not None:
                if obs_idx not in self.table:
                    self.table[obs_idx] = T.zeros((self._opponent_action_dim, self.action_dim), dtype=T.float32)
                    self.update_frequency_table[obs_idx] = T.zeros((self._opponent_action_dim, self.action_dim), dtype=T.int32)
                self.table[obs_idx][opponent_action_idx, action_idx] += + self._learning_rate * td_error
                self.update_frequency_table[obs_idx][opponent_action_idx, action_idx] += 1
            else:
                if obs_idx not in self.table:
                    self.table[obs_idx] = T.zeros(self.action_dim, dtype=T.float32)
                    self.update_frequency_table[obs_idx] = T.zeros(self.action_dim, dtype=T.int32)
                self.table[obs_idx][action_idx] += + self._learning_rate * td_error
                self.update_frequency_table[obs_idx][action_idx] += 1

    def q(self, obs: T.Tensor, action: int | None = None, opponent_action: int | None = None) -> T.Tensor:
        action_idx = slice(None) if action is None else action
        opponent_action_idx = slice(None) if opponent_action is None else opponent_action
        
        res = None
        if isinstance(self.table, T.Tensor):
            res = self.table[self._obs_idx(obs), opponent_action_idx, action_idx] if self.considering_opponent else self.table[self._obs_idx(obs), action_idx] 
        else:
            obs_idx = self._obs_idx(obs)
            if obs_idx not in self.table:
                res = self.table_empty_value[opponent_action_idx, action_idx]
            else:
                res = self.table[self._obs_idx(obs)][opponent_action_idx, action_idx] if self.considering_opponent else self.table[self._obs_idx(obs)][action_idx]
            
        return T.from_numpy(res).unsqueeze(0) if isinstance(res, np.ndarray) else T.tensor(res).unsqueeze(0)

    def update_frequency(self, obs: T.Tensor) -> T.Tensor:
        """Get the update frequency for the given observation."""
        if isinstance(self.update_frequency_table, T.Tensor):
            return self.update_frequency_table[self._obs_idx(obs), :, :] if self.considering_opponent else self.update_frequency_table[self._obs_idx(obs), :]
        else:
            obs_idx = self._obs_idx(obs)
            if obs_idx not in self.update_frequency_table:
                return T.zeros((self._opponent_action_dim, self.action_dim), dtype=T.int32) if self._opponent_action_dim is not None else T.zeros(self.action_dim, dtype=T.int32)
            return self.update_frequency_table[obs_idx][:, :] if self.considering_opponent else self.update_frequency_table[obs_idx][:]

    def save(self, path: str):
        table_path = path + "_table"
        update_frequency_table_path = path + "_update_frequency_table"
        with open(table_path, "wb") as f:
            T.save(self.table, f)
        with open(update_frequency_table_path, "wb") as f:
            T.save(self.update_frequency_table, f)
    
    def load(self, path: str):
        table_path = path + "_table"
        update_frequency_table_path = path + "_update_frequency_table"
        self.table = T.load(table_path, allow_pickle=True)
        self.update_frequency_table = T.load(update_frequency_table_path, allow_pickle=True)

    @property
    def considering_opponent(self) -> bool:
        return self._opponent_action_dim is not None

    def sparsity(self) -> float:
        """Calculate how sparse the Q-table is (i.e. how many entries are 0)."""
        if isinstance(self.table, T.Tensor):
            if self.table.size == 0:
                return 0.0
            return 1 - len(self.table.nonzero()[0]) / self.table.nelement()
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
    def opponent_action_dim(self) -> int | None:
        return self._opponent_action_dim

    @property
    def discount(self) -> float:
        return self._discount
    
    @discount.setter
    def discount(self, discount: float):
        self._discount = discount
    
    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        self._learning_rate = learning_rate


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_layer_sizes: list[int] | None = None,
        hidden_layer_activation: type[nn.Module] = nn.Identity,
        representation: nn.Module | None = None,
        is_footsies: bool = False,
        use_dense_reward: bool = True, # Dictates the range of possible output values
        opponent_action_dim: int | None = None,
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
    
    def forward(self, obs: T.Tensor):
        rep = self.representation(obs)
        qs = self.q_layers(rep)
        return qs * self._output_multiplier

    def from_representation(self, rep: T.Tensor) -> T.Tensor:
        qs = self.q_layers(rep)
        return qs * self._output_multiplier

    @property
    def obs_dim(self) -> int:
        return self._obs_dim
    
    @property
    def action_dim(self) -> int:
        return self._action_dim
    
    @property
    def opponent_action_dim(self) -> int | None:
        return self._opponent_action_dim


class QFunctionNetwork(QFunction):
    """Q-value table that uses a neural network to approximate the Q-values."""

    def __init__(
        self,
        q_network: QNetwork,
        action_dim: int,
        opponent_action_dim: int | None = None,
        discount: float = 1.0,
        learning_rate: float = 1e-2,
        target_network: QNetwork | None = None,
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
        - `target_network_update_interval`: the interval between target network updates, in terms of update steps
        """
        self._action_dim = action_dim
        self._opponent_action_dim = opponent_action_dim
        self._discount = discount
        self._target_network_update_interval = target_network_update_interval

        self.q_network = q_network
        self.q_network_optimizer = T.optim.SGD(self.q_network.parameters(), lr=learning_rate)
        if target_network is not None:
            # Make a copy of the Q-network, but with parameters set to 0 to remove any noise that the Q-network would pick up
            self.target_network = target_network
            for param in self.target_network.parameters():
                param.data.zero_()
            
            # Disable gradient computation to get speed boost.
            self.target_network.requires_grad_(False)
        else:
            self.target_network = None
        
        # Target network update tracker
        self._current_update_step = 0

    def _update_q_value(self, obs: T.Tensor, target: float, action: int | None = None, opponent_action: int | None = None):        
        action_idx = slice(None) if action is None else action
        opponent_action_idx = slice(None) if opponent_action is None else opponent_action

        self.q_network_optimizer.zero_grad()

        predicted = self.q_network(obs)
        loss = T.mean((target - predicted[:, opponent_action_idx, action_idx])**2)
        loss.backward()

        self.q_network_optimizer.step()

        self._current_update_step += 1
        if self.target_network is not None and self._current_update_step >= self._target_network_update_interval:
            LOGGER.debug("Target network has been updated")
            for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(q_param.data)
            self._current_update_step = 0
    
    # The target network is not used by default, being used only for estimating the Q-value of the next observation and avoid instability
    def q(self, obs: T.Tensor, action: int | None = None, opponent_action: int | None = None, *, use_target_network: bool = False) -> T.Tensor:
        action_idx = slice(None) if action is None else action
        opponent_action_idx = slice(None) if opponent_action is None else opponent_action

        if use_target_network:
            if self.target_network is None:
                raise ValueError("cannot use the target network since it was not requested")
            network = self.target_network
        else:
            network = self.q_network

        return network(obs)[:, opponent_action_idx, action_idx]

    def save(self, path: str):
        T.save(self.q_network.state_dict(), path)
    
    def load(self, path: str):
        state_dict = T.load(path)
        self.q_network.load_state_dict(state_dict)
        if self.target_network is not None:
            self.target_network.load_state_dict(state_dict)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def opponent_action_dim(self) -> int | None:
        return self._opponent_action_dim

    @property
    def discount(self) -> float:
        return self._discount

    @discount.setter
    def discount(self, discount: float):
        self._discount = discount

    @property
    def learning_rate(self) -> float:
        return self.q_network_optimizer.param_groups[0]["lr"]
    
    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        self.q_network_optimizer.param_groups[0]["lr"] = learning_rate

    @property
    def considering_opponent(self) -> bool:
        return self._opponent_action_dim is not None


# NOTE: this is not considering the discretzation exactly as is in the Q-table (more specifically, the move frames are not handled the same way)
class QFunctionNetworkDiscretized(QFunctionNetwork):
    """Q-value table that uses a neural network to approximate the Q-values. Observations are discretized in a similar way to the tabular Q-value estimation."""

    def __init__(
        self,
        q_network: QNetwork,
        action_dim: int,
        opponent_action_dim: int | None = None,
        discount: float = 1.0,
        learning_rate: float = 1e-2,
        target_network: QNetwork | None = None,
        target_network_update_interval: int = 1000,
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
        - `target_network`: an extra, slower-changing, target network to stabilize learning. If `None`, it won't be used
        - `target_network_update_interval`: the interval between target network updates, in terms of update steps
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
            target_network=target_network,
            target_network_update_interval=target_network_update_interval,
        )
        if environment not in ("footsies", "mountain car"):
            raise ValueError(f"environment '{environment}' not supported")

        self.environment = environment

        if self.environment == "footsies":
            self.move_frame_n_bins = move_frame_n_bins
            self.position_n_bins = position_n_bins
            self._obs_dim = self.env_obs_dim("footsies", move_frame=move_frame_n_bins, position=position_n_bins)

            # Leave some leeway for the start and end of the linear space, since the start and end are part of the observation
            self.move_frame_bins = T.linspace(-0.1, 1.1, move_frame_n_bins)
            self.position_bins = T.linspace(-1.1, 1.1, position_n_bins)

        elif self.environment == "mountain car":
            # These variables are hardcorded for the mountain car environment
            position_n_bins = 20
            velocity_n_bins = 20
            self._obs_dim = self.env_obs_dim("mountain car", position=position_n_bins, velocity=velocity_n_bins)
            self.position_n_bins = position_n_bins

            self.position_bins = T.linspace(-1.2, 0.6, position_n_bins)
            self.velocity_bins = T.linspace(-0.07, 0.07, velocity_n_bins)

    def _transform_obs(self, obs: T.Tensor) -> T.Tensor:
        """Transform the continuous environment observation into a discretized one, similar in nature to the Q-table."""
        obs = obs.squeeze()
        
        gu1, gu2 = T.round(obs[0:2] * 3).tolist()
        # We set right=True to mimic the behavior of NumPy's digitize (they are exactly opposite)
        mf1, mf2 = T.bucketize(obs[32:34], self.move_frame_bins, right=True).tolist()
        po1, po2 = T.bucketize(obs[34:36], self.position_bins, right=True).tolist()
        
        obs_discretized = T.hstack((
            # Guard
            F.one_hot(T.tensor([gu1, gu2]).long(), num_classes=4).float().flatten(),
            # Moves
            obs[2:32],
            # Move frame
            F.one_hot(T.tensor([mf1, mf2]).long(), num_classes=self.move_frame_n_bins).float().flatten(),
            # Position
            F.one_hot(T.tensor([po1, po2]).long(), num_classes=self.position_n_bins).float().flatten(),
        ))

        return obs_discretized.unsqueeze(0)

    def _update_q_value(self, obs_discretized: T.Tensor, target: float, action: int | None = None, opponent_action: int | None = None):
        obs_discretized = self._transform_obs(obs_discretized)
        super()._update_q_value(obs_discretized, target, action, opponent_action)
    
    def q(self, obs: T.Tensor, *args, **kwargs) -> float | T.Tensor:
        obs_discretized = self._transform_obs(obs)
        return super().q(obs_discretized, *args, **kwargs)

    @property
    def obs_dim(self) -> int | None:
        return self._obs_dim

    @staticmethod
    def env_obs_dim(environment: str, **kwargs) -> int | None:
        if environment == "footsies":
            move_frame = kwargs["move_frame"]
            position = kwargs["position"]
            return 4*2 + 15*2 + move_frame*2 + position*2
        elif environment == "mountain car":
            position = kwargs["position"]
            velocity = kwargs["velocity"]
            return position + velocity

        return None
