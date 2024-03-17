from torch import nn
from copy import deepcopy
from agents.action import ActionMap
from agents.the_one.agent import FootsiesAgent as TheOneAgent
from agents.a2c.agent import FootsiesAgent as A2CAgent
from agents.a2c.a2c import A2CQLearner, ActorNetwork
from agents.ql.ql import QFunctionNetwork, QNetwork
from agents.mimic.agent import PlayerModel
from agents.the_one.loggables import get_loggables


CONSIDER_OPPONENT_ACTION = True

def model_init(observation_space_size: int, action_space_size: int, *,
    actor_lr: float = 1e-1,
    critic_lr: float = 1e-2,
    actor_entropy_coef: float = 0.0001,
    critic_tanh: bool = False,
    critic_discount: float = 1.0,
    critic_agent_update: str = "expected_sarsa",
    critic_opponent_update: str = "expected_sarsa",
) -> tuple[TheOneAgent, dict[str, list]]:
    
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
        is_footsies=critic_tanh,
        use_dense_reward=False,
        opponent_action_dim=opponent_action_dim if CONSIDER_OPPONENT_ACTION else None,
    )

    target_network = deepcopy(critic_network)

    critic = QFunctionNetwork(
        q_network=critic_network,
        action_dim=action_dim,
        opponent_action_dim=opponent_action_dim,
        discount=critic_discount,
        learning_rate=critic_lr,
        target_network=target_network,
        target_network_update_interval=1000,
    )

    learner = A2CQLearner(
        actor=actor,
        critic=critic,
        actor_learning_rate=actor_lr,
        actor_entropy_loss_coef=actor_entropy_coef,
        policy_cumulative_discount=False,
        agent_update_style=getattr(A2CQLearner.UpdateStyle, critic_agent_update.upper()),
        opponent_update_style=getattr(A2CQLearner.UpdateStyle, critic_opponent_update.upper()),
    )

    a2c = A2CAgent(
        learner=learner,
        opponent_action_dim=opponent_action_dim if CONSIDER_OPPONENT_ACTION else None,
        footsies=True,
        use_opponents_perspective=False,
    )

    agent = TheOneAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        opponent_action_dim=opponent_action_dim if CONSIDER_OPPONENT_ACTION else None,
        representation=None,
        a2c=a2c,
        opponent_model=None,
        game_model=None,
        reaction_time_emulator=None,
        over_simple_actions=True,
        remove_special_moves=True,
        rollback_as_opponent_model=True,
        game_model_learning_rate=1e-4,
    )

    loggables = get_loggables(agent)

    return agent, loggables