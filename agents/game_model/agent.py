import os
import torch
import numpy as np
from torch import nn
from agents.base import FootsiesAgentBase
from agents.utils import FOOTSIES_ACTION_MOVE_INDICES_MAP
from agents.torch_utils import create_layered_network
from gymnasium import Env, Space
from typing import Callable, Tuple


class GameModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        agent_action_dim: int,
        opponent_action_dim: int,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: nn.Module = nn.LeakyReLU,
    ):
        super().__init__()
        
        input_dim = state_dim + agent_action_dim + opponent_action_dim
        
        self.layers = create_layered_network(
            input_dim, state_dim, hidden_layer_sizes, hidden_layer_activation
        )

        # Different activations for different parts of the output state
        # This is a "staircase sigmoid" for snapping to integers. It's not differentiable everywhere, but PyTorch handles this gracefully
        self.guard_activation = lambda x: 1 / (1 + torch.exp(-15 * (torch.remainder(x, 1) - 0.5))) + torch.floor(x)
        self.move_activation = nn.Softmax(dim=1)
        self.move_progress_activation = nn.Identity()
        self.position_activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.layers(x)
        # NOTE: the stacking order is important! it depends on how the environment observations are flattened (guard comes first, then move, etc.)
        return torch.hstack((
            self.guard_activation(res[:, 0:2]),
            self.move_activation(res[:, 2:17]),     # player 1
            self.move_activation(res[:, 17:32]),    # player 2
            self.move_progress_activation(res[:, 32:34]),
            self.position_activation(res[:, 34:36]),
        ))


# NOTE: trained by example
# NOTE: player 1 is assumed to be the agent
class FootsiesAgent(FootsiesAgentBase):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        opponent_action_dim: int,
        by_primitive_actions: bool = False,
        move_transition_scale: float = 60.0, # scale training examples where move transitions occur, since they are very important
        optimize_frequency: int = 1000, # mini-batch size, bad name
        learning_rate: float = 1e-2,
        hidden_layer_sizes_specification: str = "64,64",
        hidden_layer_activation_specification: str = "LeakyReLU",
    ):
        if by_primitive_actions:
            raise NotImplementedError("can't train on primitive opponent actions yet")

        self.by_primitive_actions = by_primitive_actions
        self.state_dim = observation_space.shape[0]
        self.agent_action_dim = action_space.shape[0] if by_primitive_actions else opponent_action_dim
        self.opponent_action_dim = opponent_action_dim
        self.move_transition_scale = move_transition_scale
        self.optimize_frequency = optimize_frequency
        self.learning_rate = learning_rate

        self.game_model = GameModel(
            state_dim=self.state_dim,
            agent_action_dim=self.agent_action_dim,
            opponent_action_dim=self.opponent_action_dim,
            hidden_layer_sizes=[int(n) for n in hidden_layer_sizes_specification.split(",")] if hidden_layer_sizes_specification else [],
            hidden_layer_activation=getattr(nn, hidden_layer_activation_specification),
        )

        self.optimizer = torch.optim.SGD(params=self.game_model.parameters(), lr=learning_rate)

        self.state_batch_as_list = []

        self.current_observation = None
        self.current_opponent_action = None
        self.current_agent_action = None

        self.cummulative_loss = 0
        self.cummulative_loss_n = 0

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.tensor(obs, dtype=torch.float32).reshape((1, -1))

    def _action_to_tensor(self, action: int, num_classes: int) -> torch.Tensor:
        return nn.functional.one_hot(torch.tensor(action), num_classes=num_classes).unsqueeze(0)

    def _update_batch(self, obs: np.ndarray, agent_action: int, opponent_action: int, next_obs):
        obs = self._obs_to_tensor(obs)
        next_obs = self._obs_to_tensor(next_obs)
        agent_action_oh = self._action_to_tensor(agent_action, self.agent_action_dim)
        opponent_action_oh = self._action_to_tensor(opponent_action, self.opponent_action_dim)
        self.state_batch_as_list.append(torch.hstack((obs, agent_action_oh, opponent_action_oh, next_obs)))

    def act(self, obs: np.ndarray) -> "any":
        self.current_observation = obs
        return 0

    def update(self, next_obs: np.ndarray, reward: float, terminated: bool, truncated: bool, info: dict):
        if not self.by_primitive_actions:
            # Reminder: these actions are the ones in the next observation!
            agent_action = info["p1_move"]
            opponent_action = info["p2_move"]

            agent_action = FOOTSIES_ACTION_MOVE_INDICES_MAP[agent_action]
            opponent_action = FOOTSIES_ACTION_MOVE_INDICES_MAP[opponent_action]
    
        else:
            return

        self._update_batch(self.current_observation, agent_action, opponent_action, next_obs)

        if len(self.state_batch_as_list) >= self.optimize_frequency:
            batch = torch.cat(self.state_batch_as_list)
            loss = self.train(batch)
            
            self.cummulative_loss += loss
            self.cummulative_loss_n += 1
            
            self.state_batch_as_list.clear()

    def train(self, batch: torch.Tensor) -> float:
        self.optimizer.zero_grad()

        x_size = self.state_dim + self.agent_action_dim + self.opponent_action_dim
        batch_x = batch[:, :x_size]
        batch_y = batch[:, x_size:]

        predicted = self.game_model(batch_x)
        # Euclidean distance
        guard_distance = torch.sqrt(torch.sum((predicted[:, 0:2] - batch_y[:, 0:2])**2, dim=1))
        # Cross entropy loss
        move_distance_p1 = -torch.sum(torch.log(predicted[:, 2:17]) * batch_y[:, 2:17], dim=1)
        move_distance_p2 = -torch.sum(torch.log(predicted[:, 17:32]) * batch_y[:, 17:32], dim=1)
        # Euclidean distance
        move_progress_and_position_distance = torch.sqrt(torch.sum((predicted[:, 32:36] - batch_y[:, 32:36])**2, dim=1))

        # Give more weight to training examples in which a move transition occurred
        # NOTE: this assumes the state variables are placed before the agent and opponent action features
        move_transition_multiplier = 1 + torch.any(batch_x[:, 2:32] != batch_y[:, 2:32], dim=1) * (self.move_transition_scale - 1)

        loss = torch.mean((guard_distance + move_distance_p1 + move_distance_p2 + move_progress_and_position_distance) * move_transition_multiplier)
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def evaluate_average_loss(self, *, clear: bool) -> float:
        res = (
            self.cummulative_loss / self.cummulative_loss_n
        ) if self.cummulative_loss_n != 0 else 0
        
        if clear:
            self.cummulative_loss = 0
            self.cummulative_loss_n = 0

        return res

    def predict(self, obs: np.ndarray, agent_action: int, opponent_action: int) -> np.ndarray:
        """Predict the next observation. The prediction is sanitized to contain valid values"""
        with torch.no_grad():
            next_obs: torch.Tensor = self.game_model(torch.hstack((
                self._obs_to_tensor(obs),
                self._action_to_tensor(agent_action, self.agent_action_dim),
                self._action_to_tensor(opponent_action, self.opponent_action_dim),
            )))

        # Get the maximum
        next_obs[:, 2:17] = 1.0 * (next_obs[:, 2:17] == torch.max(next_obs[:, 2:17]))
        next_obs[:, 17:32] = 1.0 * (next_obs[:, 17:32] == torch.max(next_obs[:, 17:32]))
        next_obs[:, 32] = torch.clamp(next_obs[:, 32], 0.0, 1.0)
        next_obs[:, 33] = torch.clamp(next_obs[:, 33], 0.0, 1.0)

        return next_obs.numpy(force=True)

    def load(self, folder_path: str):
        model_path = os.path.join(folder_path, "model_weights.pth")
        self.game_model.load_state_dict(torch.load(model_path))

    def save(self, folder_path: str):
        model_path = os.path.join(folder_path, "model_weights.pth")
        torch.save(self.game_model.state_dict(), model_path)        

    def extract_policy(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        return lambda s: None