import numpy as np
import torch
from torch import nn
from agents.torch_utils import create_layered_network
from intrinsic.base import IntrinsicRewardScheme
from agents.action import ActionMap


class AbstractEnvironmentEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        encoded_dim: int,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: type[nn.Module] = nn.Identity,
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
        hidden_layer_activation: type[nn.Module] = nn.Identity,
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
        hidden_layer_activation: type[nn.Module] = nn.Identity,
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

            intrinsic_reward = self.reward_scale * 0.5 * (predicted_encoded_next_obs - encoded_next_obs).pow(2).mean()

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


class ICMScheme(IntrinsicRewardScheme):
    def __init__(
        self,
        trainer: IntrinsicCuriosityTrainer,
    ):
        self.trainer = trainer
        self.icm = trainer.curiosity

        
    def _append_opponent_action(self, obs: torch.Tensor, opponent_action: int) -> torch.Tensor:
        opponent_action_oh = nn.functional.one_hot(torch.tensor([opponent_action]), num_classes=ActionMap.n_simple()).float()
        return torch.hstack((obs, opponent_action_oh))

    def update_and_reward(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict) -> float:
        p1_action, p2_action = ActionMap.simples_from_torch_transition(obs, next_obs)

        # Include the opponent's future action in the observation, which should help in determining the future state
        obs_with_opp = self._append_opponent_action(obs, p2_action)

        self.trainer.train(obs_with_opp, p1_action, next_obs)

        return self.icm.intrinsic_reward(obs_with_opp, p1_action, next_obs)

    @staticmethod
    def basic(obs_dim: int = 36, encoded_dim: int = 16) -> "IntrinsicRewardScheme":
        encoder = AbstractEnvironmentEncoder(
            obs_dim=obs_dim,
            encoded_dim=encoded_dim,
            hidden_layer_sizes=[128, 128],
            hidden_layer_activation=nn.LeakyReLU,
        )
        
        icm = IntrinsicCuriosityModule(
            encoder=encoder,
            inverse_model=InverseEnvironmentModel(
                encoded_dim=encoded_dim,
                action_dim=ActionMap.n_simple(),
                encoder=encoder,
                hidden_layer_sizes=[64, 64],
                hidden_layer_activation=nn.LeakyReLU,
            ),
            forward_model=ForwardEnvironmentModel(
                encoded_dim=encoded_dim,
                action_dim=ActionMap.n_simple(),
                encoder=encoder,
                hidden_layer_sizes=[64, 64],
                hidden_layer_activation=nn.LeakyReLU,
            ),
            reward_scale=1.0,
        )

        trainer = IntrinsicCuriosityTrainer(
            curiosity=icm,
            optimizer=torch.optim.SGD,
            learning_rate=1e-3,
            beta=0.2,
        )

        return ICMScheme(trainer=trainer)
