import torch
from torch import nn
from agents.torch_utils import create_layered_network


# A recurrent architecture might be needed for the FOOTSIES environment, as there is useful information by considering history (e.g. determining whether a dash/backdash will occur)
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
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.layers(obs)
    

class AbstractGameModel(nn.Module):
    def __init__(
        self,
        action_dim: int,
        opponent_action_dim: int,
        obs_dim: int,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: nn.Module = nn.Identity,
        representation: nn.Module = None,
    ):
        """
        Module for predicting the next abstract environment representation from the current representation.

        Parameters
        ----------
        - `action_dim`: size of the agent's action space
        - `opponent_action_dim`: size of the opponent's action space
        - `representation_dim`: size of the hidden representation
        - `hidden_layer_sizes`: list of the sizes of the hidden layers. If None, no hidden layers will be created
        - `hidden_layer_activation`: the activation function that will be used on all hidden layers
        - `representation`: representation module that will be used to convert the input into a hidden representation. If None, no such module is inserted
        """
        super().__init__()

        self.layers = create_layered_network(action_dim + opponent_action_dim + obs_dim, obs_dim, hidden_layer_sizes, hidden_layer_activation)
        self.representation = nn.Identity() if representation is None else representation
    
    def forward(self, obs: torch.Tensor, agent_action_onehot: torch.Tensor, opponent_action: torch.Tensor) -> torch.Tensor:
        obs_representation = self.representation(obs)

        x = torch.hstack((obs_representation, agent_action_onehot, opponent_action))

        return self.layers(x)


class AbstractOpponentModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        opponent_action_dim: int,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: nn.Module = nn.Identity,
        representation: nn.Module = None,
    ):
        """
        Module for predicting the opponent's action given the current abstract environment representation.

        Parameters
        ----------
        - `representation_dim`: size of the hidden representation
        - `hidden_layer_sizes`: list of the sizes of the hidden layers. If None, no hidden layers will be created
        - `hidden_layer_activation`: the activation function that will be used on all hidden layers
        - `representation`: representation module that will be used to convert the input into a hidden representation. If None, no such module is inserted
        """
        super().__init__()

        self.layers = create_layered_network(obs_dim, opponent_action_dim, hidden_layer_sizes, hidden_layer_activation)
        self.layers.append(nn.Softmax(dim=1))
        if representation is not None:
            self.layers.insert(0, representation)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.layers(obs)
