import os
import torch
from agents.base import FootsiesAgentBase
from gymnasium import Env, Space
from typing import Callable, Tuple
from torch import nn


class Normalize(nn.Module):
    def forward(self, x: torch.Tensor):
        return x / torch.norm(x, p=2)


class Autoencoder(nn.Module):
    def __init__(
            self,
            state_dim: int,
            encoded_dim: int,
            normalized: bool = False,
            encoder_hidden_layers: int = 1,
            encoder_hidden_layers_size: int = 32,
            decoder_hidden_layers: int = 1,
            decoder_hidden_layer_size: int = 32,
    ):
        super().__init__()
        if encoder_hidden_layers > 0:
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, encoder_hidden_layers_size),
                *(nn.Linear(encoder_hidden_layers_size, encoder_hidden_layers_size) for _ in range(encoder_hidden_layers - 1)),
                nn.Linear(encoder_hidden_layers_size, encoded_dim)
            )

        elif encoder_hidden_layers < 0:
            self.encoder = nn.Identity()
        
        else:
            self.encoder = nn.Sequential(nn.Linear(state_dim, encoded_dim))

        if normalized:
            self.encoder.append(Normalize())

        if decoder_hidden_layers > 0:
            self.decoder = nn.Sequential(
                nn.Linear(encoded_dim, decoder_hidden_layer_size),
                *(nn.Linear(decoder_hidden_layer_size, decoder_hidden_layer_size) for _ in range(encoder_hidden_layers - 1)),
                nn.Linear(decoder_hidden_layer_size, state_dim)
            )

        elif decoder_hidden_layers < 0:
            self.decoder = nn.Identity()
        
        else:
            self.decoder = nn.Sequential(nn.Linear(encoded_dim, state_dim))

    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))


class AutoencoderAgent(FootsiesAgentBase):
    def __init__(
        self,
        observation_space_size: int,
        action_space_size: int,
        encoded_dim: int,
        normalized: bool = False,
        encoder_hidden_layers: int = 1,
        encoder_hidden_layer_size: int = 32,
        decoder_hidden_layers: int = 1,
        decoder_hidden_layer_size: int = 32,
        include_sequentiality_loss: bool = False,
        optimize_frequency: int = 1000,
        learning_rate: float = 1e-2,
    ):
        self.optimize_frequency = optimize_frequency
        self.include_sequentiality_loss = include_sequentiality_loss

        self.autoencoder = Autoencoder(
            state_dim=observation_space_size,
            encoded_dim=encoded_dim,
            normalized=normalized,
            encoder_hidden_layers=encoder_hidden_layers,
            encoder_hidden_layers_size=encoder_hidden_layer_size,
            decoder_hidden_layers=decoder_hidden_layers,
            decoder_hidden_layer_size=decoder_hidden_layer_size,
        )

        self.optimizer = torch.optim.SGD(params=self.autoencoder.parameters(), lr=learning_rate)

        self.state_batch_as_list = []

        self.cummulative_loss = 0
        self.cummulative_loss_seq = 0
        self.cummulative_loss_n = 0

    def act(self, obs, info: dict) -> "any":
        if len(self.state_batch_as_list) == 0:
            self._update_batch(obs)
        return 0

    def update(self, obs, next_obs, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict):
        self._update_batch(next_obs)

        if len(self.state_batch_as_list) >= self.optimize_frequency:
            batch = torch.cat(self.state_batch_as_list)
            loss, loss_seq = self.train(batch)
            
            self.cummulative_loss += loss
            self.cummulative_loss_seq += loss_seq
            self.cummulative_loss_n += 1
            
            self.state_batch_as_list.clear()

    def encode(self, obs) -> torch.Tensor:
        obs = self._obs_to_tensor(obs)
        with torch.no_grad:
            return self.autoencoder.encoder(obs)

    def train(self, batch: torch.Tensor) -> Tuple[float, float]:
        self.optimizer.zero_grad()

        z = self.autoencoder.encoder(batch)
        predicted = self.autoencoder.decoder(z)
        # Euclidean distance
        prediction_distance = torch.sqrt(torch.sum((predicted - batch)**2, dim=1))
        loss = torch.mean(prediction_distance)
        
        loss_seq = None
        # Assumes the vectors are normalized
        if self.include_sequentiality_loss:
            loss_seq = 1 - z[1:, :].T @ z[:-1, :]
            loss += loss_seq

        loss.backward()

        # weight: torch.Tensor = self.autoencoder.encoder[0].weight
        # weight_grad: torch.Tensor = weight.grad
        # weight_target = torch.eye(36) # we should learn the identity function
        # weight_correct_direction = torch.abs(weight_target - (weight + weight_grad)) < torch.abs(weight_target - weight)
        # bias: torch.Tensor = self.autoencoder.encoder[0].bias
        # bias_grad = bias.grad
        # bias_correct_direction = torch.abs(bias + bias_grad) < torch.abs(bias) # we should learn 0s for bias

        self.optimizer.step()

        return loss.item(), (loss_seq.item() if loss_seq is not None else 0)

    def evaluate_average_loss(self, *, clear: bool) -> Tuple[float, float]:
        res = (
            self.cummulative_loss / self.cummulative_loss_n,
            self.cummulative_loss_seq / self.cummulative_loss_n,
        ) if self.cummulative_loss_n != 0 else (0, 0)
        
        if clear:
            self.cummulative_loss = 0
            self.cummulative_loss_seq = 0
            self.cummulative_loss_n = 0

        return res

    def _obs_to_tensor(self, obs):
        return torch.tensor(obs, dtype=torch.float32).reshape((1, -1))

    def _update_batch(self, obs):
        obs = self._obs_to_tensor(obs)
        self.state_batch_as_list.append(obs)

    def load(self, folder_path: str):
        model_path = os.path.join(folder_path, "model_weights.pth")
        self.autoencoder.load_state_dict(torch.load(model_path))

    def save(self, folder_path: str):
        model_path = os.path.join(folder_path, "model_weights.pth")
        torch.save(self.autoencoder.state_dict(), model_path)

    def extract_policy(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        return lambda s: None