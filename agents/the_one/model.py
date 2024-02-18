import torch
from torch import nn
from torch_utils import create_layered_network


# NOTE: I don't think a recurrent architecture is needed for the FOOTSIES environment, there is no more useful information by considering history
class RepresentationModule(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        opponent_action_dim: int,
        representation_dim: int,
        recurrent: bool = False,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: nn.Module = nn.Identity,
    ):
        """
        Module for converting observational input into an abstract representation useful for learning other components.
        See chapter 17.3 of Reinforcement Learning: An Introduction, by Sutton & Barto, for the motivation behind this module.

        Parameters
        ----------
        - `obs_dim`: size of the environment observations
        - `action_dim`: size of the agent's action space
        - `opponent_action_dim`: size of the opponent's action space
        - `representation_dim`: size of the hidden representation, that will be learned
        - `recurrent`: whether the achitecture is recurrent, using previous representations to determine the current one
        - `hidden_layer_sizes`: list of the sizes of the hidden layers. If None, no hidden layers will be created
        - `hidden_layer_activation`: the activation function that will be used on all hidden layers
        """
        super().__init__()

        if recurrent:
            raise ValueError("a recurrent architecture is not supported")

        self.layers = create_layered_network(obs_dim + action_dim + opponent_action_dim, representation_dim, hidden_layer_sizes, hidden_layer_activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    

class AbstractGameModel(nn.Module):
    def __init__(
        self,
        representation_dim: int,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: nn.Module = nn.Identity,
    ):
        """
        Module for predicting the next abstract environment representation from the current representation.

        Parameters
        ----------
        - `representation_dim`: size of the hidden representation
        - `hidden_layer_sizes`: list of the sizes of the hidden layers. If None, no hidden layers will be created
        - `hidden_layer_activation`: the activation function that will be used on all hidden layers
        """
        super().__init__()

        self.layers = create_layered_network(representation_dim, representation_dim, hidden_layer_sizes, hidden_layer_activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AbstractOpponentModel(nn.Module):
    def __init__(
        self,
        representation_dim: int,
        opponent_action_dim: int,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: nn.Module = nn.Identity,
    ):
        """
        Module for predicting the opponent's action given the current abstract environment representation.

        Parameters
        ----------
        - `representation_dim`: size of the hidden representation
        - `hidden_layer_sizes`: list of the sizes of the hidden layers. If None, no hidden layers will be created
        - `hidden_layer_activation`: the activation function that will be used on all hidden layers
        """
        super().__init__()

        self.layers = create_layered_network(representation_dim, opponent_action_dim, hidden_layer_sizes, hidden_layer_activation)
        self.layers.append(nn.Softmax(dim=1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
