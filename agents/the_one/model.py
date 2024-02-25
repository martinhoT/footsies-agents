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

        self.representation_layers = create_layered_network(obs_dim + action_dim + opponent_action_dim, representation_dim, hidden_layer_sizes, hidden_layer_activation)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.representation_layers(obs)
    

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
        - `obs_dim`: size of the observation, or the hidden representation if one is used
        - `hidden_layer_sizes`: list of the sizes of the hidden layers. If None, no hidden layers will be created
        - `hidden_layer_activation`: the activation function that will be used on all hidden layers
        - `representation`: representation module that will be used to convert the input into a hidden representation. If None, no such module is inserted
        """
        super().__init__()

        self.game_model_layers = create_layered_network(action_dim + opponent_action_dim + obs_dim, obs_dim, hidden_layer_sizes, hidden_layer_activation)
        self.representation = nn.Identity() if representation is None else representation
    
    def forward(self, obs: torch.Tensor, agent_action_onehot: torch.Tensor, opponent_action_onehot: torch.Tensor) -> torch.Tensor:
        obs_representation = self.representation(obs)

        x = torch.hstack((obs_representation, agent_action_onehot, opponent_action_onehot))

        return self.game_model_layers(x)

    def from_representation(self, rep: torch.Tensor, agent_action_onehot: torch.Tensor, opponent_action_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.hstack((rep, agent_action_onehot, opponent_action_onehot))
        
        return self.game_model_layers(x)


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
        - `obs_dim`: size of the observation, or the hidden representation if one is used
        - `hidden_layer_sizes`: list of the sizes of the hidden layers. If None, no hidden layers will be created
        - `hidden_layer_activation`: the activation function that will be used on all hidden layers
        - `representation`: representation module that will be used to convert the input into a hidden representation. If None, no such module is inserted
        """
        super().__init__()

        self.opponent_model_layers = create_layered_network(obs_dim, opponent_action_dim, hidden_layer_sizes, hidden_layer_activation)
        self.opponent_model_layers.append(nn.Softmax(dim=1))
        self.representation = nn.Identity() if representation is None else representation
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        rep = self.representation(obs)
        return self.opponent_model_layers(rep)

    def from_representation(self, rep: torch.Tensor) -> torch.Tensor:
        return self.opponent_model_layers(rep)


# What model to report with the `model` property of the agent, since it implements FootsiesAgentTorch
class FullModel(nn.Module):
    def __init__(
        self,
        game_model: AbstractGameModel,
        opponent_model: AbstractOpponentModel,
        actor: nn.Module,
        critic: nn.Module,
    ):
        super().__init__()

        self.game_model = game_model
        self.opponent_model = opponent_model
        self.actor = actor
        self.critic = critic
    
    def forward(self, obs: torch.Tensor, agent_action_onehot: torch.Tensor, opponent_action_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        next_obs_representation = self.game_model(obs, agent_action_onehot, opponent_action_onehot)
        opponent_action_probabilities = self.opponent_model(obs)
        agent_action_probabilities = self.actor(obs)
        obs_value = self.critic(obs)

        return next_obs_representation, opponent_action_probabilities, agent_action_probabilities, obs_value
        