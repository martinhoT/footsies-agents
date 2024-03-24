from torch import nn
from copy import deepcopy
from agents.action import ActionMap
from agents.mimic.loggables import get_loggables
from agents.mimic.mimic import PlayerModel, PlayerModelNetwork, ScarStore
from agents.mimic.agent import FootsiesAgent as MimicAgent


def model_init(observation_space_size: int, action_space_size: int, *,
    dynamic_loss_weights: bool = True,
    learning_rate: float = 1e-2,
    scar_size: int = 1000,
    scar_min_loss: float = 0.1,
) -> tuple[MimicAgent, dict[str, list]]:

    obs_dim = observation_space_size
    action_dim = ActionMap.n_simple()

    p1_model = PlayerModel(
        player_model_network=PlayerModelNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            use_sigmoid_output=False,
            input_clip=False,
            input_clip_leaky_coef=0.02,
            hidden_layer_sizes=[64, 64],
            hidden_layer_activation=nn.LeakyReLU,
        ),
        scar_store=ScarStore(
            obs_dim=obs_dim,
            max_size=scar_size,
            min_loss=scar_min_loss,
        ),
        learning_rate=learning_rate,
        loss_dynamic_weights=dynamic_loss_weights,
    )

    p2_model = deepcopy(p1_model)

    agent = MimicAgent(
        action_dim=action_dim,
        by_primitive_actions=False,
        p1_model=p1_model,
        p2_model=p2_model,
    )

    loggables = get_loggables(agent)

    return agent, loggables