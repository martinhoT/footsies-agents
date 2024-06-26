from torch import nn
from agents.action import ActionMap
from agents.game_model.agent import GameModelAgent
from agents.game_model.game_model import GameModel, GameModelNetwork
from agents.game_model.loggables import get_loggables
from typing import Any


def model_init(observation_space_size: int, action_space_size: int, *,
    # Important modifiers
    residual: bool = True,
    by_differences: bool = False,
    discrete_conversion: bool = False,
    
    # Highly-tunable hyperparameters
    learning_rate: float = 1e-2,
    
    # Probably should be kept as-is
    discrete_guard: bool = False,
    remove_special_moves: bool = False,
) -> tuple[GameModelAgent, dict[str, list[Any]]]:

    obs_dim = observation_space_size
    action_dim = ActionMap.n_simple() - (2 if remove_special_moves else 0)
    opponent_action_dim = ActionMap.n_simple()

    game_model = GameModel(
        game_model_network=GameModelNetwork(
            obs_dim=obs_dim,
            p1_action_dim=action_dim,
            p2_action_dim=opponent_action_dim,
            hidden_layer_sizes=[64, 64],
            hidden_layer_activation=nn.LeakyReLU,
            residual=residual,
        ),
        learning_rate=learning_rate,
        discrete_conversion=discrete_conversion,
        discrete_guard=discrete_guard,
        by_differences=by_differences,
        epoch_timesteps=1,
        epoch_epochs=1,
        epoch_minibatch_size=1,
    )
    agent = GameModelAgent(
        game_model,
    )

    loggables = get_loggables(agent)

    return agent, loggables
