from torch import nn
from copy import deepcopy
from agents.action import ActionMap
from agents.the_one.agent import FootsiesAgent as TheOneAgent
from agents.a2c.agent import FootsiesAgent as A2CAgent
from agents.a2c.a2c import A2CQLearner, ActorNetwork
from agents.ql.ql import QFunctionNetwork, QFunctionNetworkDiscretized, QNetwork, QFunctionTable
from agents.mimic.agent import PlayerModel
from agents.the_one.loggables import get_loggables


CONSIDER_OPPONENT_ACTION = True

def model_init(observation_space_size: int, action_space_size: int, *,
    discount: float = 1.0,
    critic_tanh: bool = True,
    discretized: bool = False,
    qtable: bool = False,
    use_target_network: bool = True,
) -> tuple[TheOneAgent, dict[str, list]]:
    
    obs_dim = observation_space_size
    action_dim = ActionMap.n_simple()

    actor = ActorNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layer_sizes=[64, 64],
        hidden_layer_activation=nn.LeakyReLU,
        opponent_action_dim=action_dim if CONSIDER_OPPONENT_ACTION else None,
    )

    if qtable:
        critic = QFunctionTable(
            action_dim=action_dim,
            opponent_action_dim=action_dim if CONSIDER_OPPONENT_ACTION else None,
            discount=discount,
            learning_rate=1e-2,
            table_as_matrix=False,
            move_frame_n_bins=5,
            position_n_bins=5,
            environment="footsies",
        )

    else:
        critic_network = QNetwork(
            obs_dim=obs_dim if not discretized else QFunctionNetworkDiscretized.env_obs_dim("footsies", move_frame=5, position=5),
            action_dim=action_dim,
            hidden_layer_sizes=[128, 128],
            hidden_layer_activation=nn.LeakyReLU,
            # Will setup Tanh with appropriate range
            is_footsies=critic_tanh,
            use_dense_reward=True,
            opponent_action_dim=action_dim if CONSIDER_OPPONENT_ACTION else None,
        )

        target_network = deepcopy(critic_network)

        if discretized:
            critic = QFunctionNetworkDiscretized(
                q_network=critic_network,
                action_dim=action_dim,
                opponent_action_dim=action_dim,
                discount=discount,
                learning_rate=1e-2,
                target_network=None if not use_target_network else target_network,
                target_network_update_interval=1000,

                move_frame_n_bins=5,
                position_n_bins=5,
                environment="footsies",
            )

        else:
            critic = QFunctionNetwork(
                q_network=critic_network,
                action_dim=action_dim,
                opponent_action_dim=action_dim,
                discount=discount,
                learning_rate=1e-2,
                target_network=None if not use_target_network else target_network,
                target_network_update_interval=1000,
            )

    learner = A2CQLearner(
        actor=actor,
        critic=critic,
        actor_learning_rate=1e-2,
        actor_entropy_loss_coef=0.0,
        policy_cumulative_discount=False,
        agent_update_style=A2CQLearner.UpdateStyle.EXPECTED_SARSA,
    )

    a2c = A2CAgent(
        learner=learner,
        opponent_action_dim=action_dim if CONSIDER_OPPONENT_ACTION else None,
        footsies=True,
        use_opponents_perspective=False,
    )

    agent = TheOneAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        opponent_action_dim=action_dim if CONSIDER_OPPONENT_ACTION else None,
        representation=None,
        a2c=a2c,
        opponent_model=None,
        game_model=None,
        reaction_time_emulator=None,
        over_simple_actions=True,
        game_model_learning_rate=1e-4,
    )

    loggables = get_loggables(agent)

    return agent, loggables