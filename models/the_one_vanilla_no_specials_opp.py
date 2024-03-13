from torch import nn
from copy import deepcopy
from agents.action import ActionMap
from agents.the_one.agent import FootsiesAgent as TheOneAgent
from agents.a2c.agent import FootsiesAgent as A2CAgent
from agents.a2c.a2c import A2CQLearner, ActorNetwork
from agents.ql.ql import QFunctionNetwork, QNetwork
from agents.mimic.agent import PlayerModel, PlayerModelNetwork
from agents.the_one.loggables import get_loggables


CONSIDER_OPPONENT_ACTION = True

def model_init(observation_space_size: int, action_space_size: int, **kwargs) -> tuple[TheOneAgent, dict[str, list]]:
    obs_dim = observation_space_size
    action_dim = ActionMap.n_simple() - 2
    opponent_action_dim = ActionMap.n_simple()

    actor = ActorNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layer_sizes=[64, 64],
        hidden_layer_activation=nn.LeakyReLU,
        opponent_action_dim=opponent_action_dim if CONSIDER_OPPONENT_ACTION else None,
    )

    critic_network = QNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layer_sizes=[128, 128],
        hidden_layer_activation=nn.LeakyReLU,
        # Will setup Tanh with appropriate range
        is_footsies=True,
        use_dense_reward=True,
        opponent_action_dim=opponent_action_dim if CONSIDER_OPPONENT_ACTION else None,
    )

    target_network = deepcopy(critic_network)

    critic = QFunctionNetwork(
        q_network=critic_network,
        action_dim=action_dim,
        opponent_action_dim=opponent_action_dim,
        discount=1.0,
        learning_rate=1e-2,
        target_network=target_network,
        target_network_update_interval=1000,
    )

    learner = A2CQLearner(
        actor=actor,
        critic=critic,
        actor_learning_rate=1e-2,
        actor_entropy_loss_coef=0.1,
        policy_cumulative_discount=False,
        update_style="expected-sarsa",
    )

    a2c = A2CAgent(
        learner=learner,
        opponent_action_dim=opponent_action_dim if CONSIDER_OPPONENT_ACTION else None,
        footsies=True,
        use_opponents_perspective=False,
    )

    opponent_model = PlayerModel(
        player_model_network=PlayerModelNetwork(
            input_dim=obs_dim,
            output_dim=opponent_action_dim,
            use_sigmoid_output=False,
            input_clip=False,
            input_clip_leaky_coef=0.0,
            hidden_layer_sizes=[64, 64],
            hidden_layer_activation=nn.LeakyReLU,
        ),
        move_transition_scale=1.0,
        learning_rate=1e-2,
        reinforce_max_loss=0,
        reinforce_max_iters=1,
        scar_max_size=1000,
        scar_loss_coef=1.0,
        scar_recency_coef=0.0,
        scar_detection_threshold=float("+inf"),
        smoothed_loss_coef=0.0,
    )

    agent = TheOneAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        opponent_action_dim=opponent_action_dim if CONSIDER_OPPONENT_ACTION else None,
        representation=None,
        a2c=a2c,
        opponent_model=opponent_model,
        game_model=None,
        reaction_time_emulator=None,
        over_simple_actions=True,
        remove_special_moves=True,
        game_model_learning_rate=1e-4,
        opponent_model_learning_rate=1e-4,
    )

    loggables = get_loggables(agent)

    return agent, loggables