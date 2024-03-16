import torch
from torch import nn
from agents.torch_utils import create_layered_network


class RNDNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: type[nn.Module] = nn.Identity,
        representation: nn.Module = None,
    ):
        super().__init__()

        self.layers = create_layered_network(obs_dim, 1, hidden_layer_sizes, hidden_layer_activation)
        self.representation = nn.Identity() if representation is None else representation

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        rep = self.representation(obs)
        return self.layers(rep)

    def from_representation(self, rep: torch.Tensor) -> torch.Tensor:
        return self.layers(rep)


class RandomNetworkDistillation:
    def __init__(
        self,
        embedding_dim: int,
        learning_rate: float = 1e-2,
        reward_normalization: bool = True,
        observation_normalization: bool = True,
    ):
        self._reward_normalization = reward_normalization
        self._observation_normalization = observation_normalization

        self._target = RNDNetwork(...)
        self._predictor = RNDNetwork(...)

        # Initialize the target network randomly
        for param in self._target.parameters():
            nn.init.uniform_(param.data, -1.0, 1.0)

        self._optimizer = torch.optim.SGD(self._predictor.parameters(), lr=learning_rate)

        self._target.requires_grad_(False)

    def reward(self, obs: torch.Tensor) -> float:
        """The reward as a function of the prediction error for the given observation, between the predictor and the target network."""
        target = self._target(obs)
        prediction = self._predictor(obs)

        return (target - prediction).pow(2).mean().item()

    def update(self, obs: torch.Tensor):
        pass