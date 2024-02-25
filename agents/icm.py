import numpy as np
import torch
from torch import nn
from agents.torch_utils import create_layered_network


class AbstractEnvironmentEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        encoded_dim: int,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: nn.Module = nn.Identity,
    ):
        super().__init__()

        self.layers = create_layered_network(obs_dim, encoded_dim, hidden_layer_sizes, hidden_layer_activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        encoded_obs = self.layers(obs)
        return encoded_obs


class InverseEnvironmentModel(nn.Module):
    def __init__(
        self,
        encoded_dim: int,
        action_dim: int,
        encoder: AbstractEnvironmentEncoder,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: nn.Module = nn.Identity,
    ):
        super().__init__()

        self.encoder = encoder
        self.layers = create_layered_network(encoded_dim * 2, action_dim, hidden_layer_sizes, hidden_layer_activation)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        encoded_obs = self.encoder(obs)
        next_encoded_obs = self.encoder(next_obs)
        logits = self.layers(torch.hstack((encoded_obs, next_encoded_obs)))
        return self.softmax(logits)


class ForwardEnvironmentModel(nn.Module):
    def __init__(
        self,
        encoded_dim: int,
        action_dim: int,
        encoder: AbstractEnvironmentEncoder,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: nn.Module = nn.Identity,
    ):
        super().__init__()

        self.encoder = encoder
        self.layers = create_layered_network(encoded_dim + action_dim, encoded_dim, hidden_layer_sizes, hidden_layer_activation)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        encoded_obs = self.encoder(obs)
        return self.layers(torch.hstack((encoded_obs, action)))


# From: https://pathak22.github.io/noreward-rl/
class IntrinsicCuriosityModule(nn.Module):
    def __init__(
        self,
        encoder: AbstractEnvironmentEncoder,
        inverse_model: InverseEnvironmentModel,
        forward_model: ForwardEnvironmentModel,
        reward_scale: float = 1.0,
    ):
        super().__init__()

        # TODO: the encoder being separate of the models, while the models use *implicitly* the encoder, is goofy
        self.encoder = encoder
        self.inverse_model = inverse_model
        self.forward_model = forward_model
        self.reward_scale = reward_scale

        self.action_dim = self.inverse_model.layers[-1].out_features

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action = nn.functional.one_hot(action, num_classes=self.action_dim).detach()

        predicted_agent_action = self.inverse_model(obs, next_obs)
        predicted_encoded_next_obs = self.forward_model(obs, action)

        return predicted_agent_action, predicted_encoded_next_obs

    def intrinsic_reward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> float:
        with torch.no_grad():
            encoded_next_obs = self.encoder(next_obs)
            _, predicted_encoded_next_obs = self(obs, action, next_obs)

            intrinsic_reward = self.reward_scale * 0.5 * torch.mean((predicted_encoded_next_obs - encoded_next_obs) ** 2)

            return intrinsic_reward.item()
    

class IntrinsicCuriosityTrainer:
    def __init__(
        self,
        curiosity: IntrinsicCuriosityModule,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        learning_rate: float = 1e-3,
        beta: float = 0.2,
    ):
        """
        Class for training an Intrinsic Curiosity Module (ICM).
        
        Parameters
        ----------
        - `curiosity`: the instrisic curiosity module to train
        - `optimizer`: the optimizer class to use for training
        - `learning_rate`: the learning rate for the optimizer
        - `beta`: the linear interpolation scale between the inverse model loss (1 - `beta`) and the forward model loss (`beta`)
        """

        self.curiosity = curiosity
        self.beta = beta

        self.inverse_model_loss_fn = nn.L1Loss()
        self.forward_model_loss_fn = nn.MSELoss()

        self.optimizer = optimizer(curiosity.parameters(), lr=learning_rate)

        # For tracking
        self.inverse_model_loss = None
        self.forward_model_loss = None
    
    def train(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor):
        self.optimizer.zero_grad()

        # Put arguments in the correct format
        encoded_next_obs = self.curiosity.encoder(next_obs).detach()
        action_onehot = nn.functional.one_hot(action, num_classes=self.curiosity.action_dim).detach()

        predicted_agent_action, predicted_encoded_next_obs = self.curiosity(obs, action, next_obs)
    
        inverse_model_loss = self.inverse_model_loss_fn(predicted_agent_action, action_onehot)
        forward_model_loss = self.forward_model_loss_fn(predicted_encoded_next_obs, encoded_next_obs)

        loss = (1 - self.beta) * inverse_model_loss + self.beta * forward_model_loss
        loss.backward()

        self.optimizer.step()
        
        self.inverse_model_loss = inverse_model_loss.item()
        self.forward_model_loss = forward_model_loss.item()


class NoveltyTable:
    def __init__(
        self,
        reward_scale: float = 1.0
    ):
        self.table = {}
        self.reward_scale = reward_scale
    
    def register(self, obs: torch.Tensor):
        o = obs.numpy().tobytes()
        self.table[o] = self.table.get(o, 0) + 1
    
    def query(self, obs: torch.Tensor) -> int:
        o = obs.numpy().tobytes()
        return self.table.get(o, 0)

    def intrinsic_reward(self, obs: torch.Tensor) -> float:
        return self.reward_scale / self.query(obs)