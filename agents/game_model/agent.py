import os
import torch
import numpy as np
from torch import nn
from agents.base import FootsiesAgentBase
from gymnasium import Env, Space
from typing import Callable, Tuple
from footsies_gym.moves import actionable_moves


class GameModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        agent_action_dim: int,
        opponent_action_dim: int,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + agent_action_dim + opponent_action_dim, state_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


# NOTE: trained by example
# NOTE: player 1 is assumed to be the agent
class FootsiesAgent(FootsiesAgentBase):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        opponent_action_dim: int,
        by_primitive_actions: bool = False,
        optimize_frequency: int = 1000,
        learning_rate: float = 1e-2,
    ):
        if by_primitive_actions:
            raise NotImplementedError("can't train on primitive opponent actions yet")

        self.by_primitive_actions = by_primitive_actions
        self.state_dim = observation_space.shape[0]
        self.agent_action_dim = action_space.shape[0] if by_primitive_actions else opponent_action_dim
        self.opponent_action_dim = opponent_action_dim
        self.optimize_frequency = optimize_frequency
        self.learning_rate = learning_rate

        self.game_model = GameModel(
            state_dim=self.state_dim,
            agent_action_dim=self.agent_action_dim,
            opponent_action_dim=self.opponent_action_dim,
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
    
    # NOTE: DAMAGE is being included in the actions
    def simplify_action(self, action: int) -> int:
        """Simplify the player's action. For instance, guard motions will be set to BACKWARD"""
        if action >= 10 and action <= 14: # GUARD_M ... GUARD_PROXIMITY
            action = 2 # BACKWARD
        
        return action

    def act(self, obs: np.ndarray) -> "any":
        self.current_observation = obs
        return 0

    def update(self, next_obs: np.ndarray, reward: float, terminated: bool, truncated: bool, info: dict):
        if not self.by_primitive_actions:
            agent_action = info["p1_move"]
            opponent_action = info["p2_move"]

            agent_action = self.simplify_action(agent_action)
            opponent_action = self.simplify_action(opponent_action)
    
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
        prediction_distance = torch.sqrt(torch.sum((predicted - batch_y)**2, dim=1))
        loss = torch.mean(prediction_distance)

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def evaluate_average_loss(self, *, clear: bool) -> float:
        res = (
            self.cummulative_loss / self.cummulative_loss_n
        ) if self.cummulative_loss_n != 0 else (0, 0)
        
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

        next_obs[:, 2:17] = 1.0 * (next_obs[:, 2:17] > 0.5)
        next_obs[:, 17:32] = 1.0 * (next_obs[:, 17:32] > 0.5)
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