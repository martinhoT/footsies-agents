from copy import deepcopy
import os
import torch
from torch import nn
from agents.base import FootsiesAgentBase
from gymnasium import Env
from typing import Callable, Tuple
from gymnasium import Space
from footsies_gym.moves import FootsiesMove


RELEVANT_MOVES = set(FootsiesMove) - {FootsiesMove.WIN, FootsiesMove.DEAD}


class PlayerModelNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, output_sigmoid: bool = False):
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            # nn.Linear(32, 32),
            nn.Linear(32, output_dim),
            nn.Sigmoid(output_dim)
            if output_sigmoid
            else nn.Softmax(output_dim),
        )
    
    def forward(self, x: torch.Tensor):
        return self.layers(x)


class PlayerModel:
    def __init__(self, n_moves: int, optimize_frequency: int = 1000):
        self.optimize_frequency = optimize_frequency
        
        self.network = PlayerModelNetwork(n_moves)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(lr=1e-3)

        self.batch_as_list = []
        self.step = 0
        self.cummulative_loss = 0
        self.cummulative_loss_n = 0
    
    def update(self, obs: torch.Tensor, action: torch.Tensor):
        self.batch_as_list.append(torch.cat([obs, action]))
        self.step += 1

        if self.step >= self.optimize_frequency:
            batch = torch.stack(self.batch_as_list)
            error = None

            # Keep training on examples that are problematic
            while error is None or error > 0:
                self.optimizer.zero_grad()
                
                predicted = self.network(batch[:, :-1])
                loss = self.loss_function(predicted, batch[:, -1])
                
                loss.backward()
                self.optimizer.step()

                # TODO: don't try this for now
                # error = (loss > ...).sum()
                error = 0

                self.cummulative_loss += loss.sum()
                self.cummulative_loss_n += loss.size(0)
            
            self.batch_as_list.clear()


class FootsiesAgent(FootsiesAgentBase):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        over_primitive_actions: bool = False,
        optimize_frequency: int = 1000,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.over_primitive_actions = over_primitive_actions

        self.n_moves = len(RELEVANT_MOVES) if over_primitive_actions else len(action_space)

        self.p1_model = PlayerModel(self.n_moves, optimize_frequency)
        self.p2_model = PlayerModel(self.n_moves, optimize_frequency)

        self.current_observation = None

    def act(self, obs) -> "any":
        self.current_observation = obs
        with torch.no_grad():
            out = (self.p1_model.predict(obs), self.p2_model.predict(obs))
        return out

    def update(self, next_obs, reward: float, terminated: bool, truncated: bool, info: dict):
        key = "action" if self.over_primitive_actions else "move"
        # This is assuming that, when using moves, all relevant moves have contiguous IDs from 0, without breaks
        # That is, the irrelevant moves (WIN and DEAD) have the largest IDs
        p1_move = self._move_onehot(info["p1_" + key])
        p2_move = self._move_onehot(info["p2_" + key])

        self.p1_model.update(self.current_observation, p1_move)
        self.p2_model.update(self.current_observation, p2_move)

    def _move_onehot(self, move: int):
        onehot = torch.zeros((1, self.n_moves))
        onehot[0, move] = 1
        return onehot

    # def preprocess(self, env: Env):
    #     ...

    def load(self, folder_path: str):
        p1_path = os.path.join(folder_path, "p1")
        p2_path = os.path.join(folder_path, "p2")
        self.p1_model.save(p1_path)
        self.p2_model.save(p2_path)
        
    def save(self, folder_path: str):
        p1_path = os.path.join(folder_path, "p1")
        p2_path = os.path.join(folder_path, "p2")
        self.p1_model.save(p1_path)
        self.p2_model.save(p2_path)

    def evaluate_average_loss_and_clear(self, p1: bool) -> float:
        model = self.p1_model if p1 else self.p2_model

        res = (
            (model.cummulative_loss / model.cummulative_loss_n)
            if model.cummulative_loss_n != 0
            else 0
        )
        
        model.cummulative_loss = 0
        model.cummulative_loss_n = 0
        return res

    # NOTE: this only works if the class was defined to be over primitive actions
    def extract_policy(self, env: Env, use_p1: bool = True) -> Callable[[dict], Tuple[bool, bool, bool]]:
        model_to_use = self.p1_model if use_p1 else self.p2_model

        policy_network = deepcopy(model_to_use.network)
        policy_network.requires_grad_(False)

        def internal_policy(obs):
            obs = self._obs_to_torch(obs)
            predicted = policy_network(obs)
            return torch.argmax(predicted)

        return super()._extract_policy(env, internal_policy)