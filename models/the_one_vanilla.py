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

def model_init(observation_space_size: int, action_space_size: int, **kwargs) -> tuple[TheOneAgent, dict[str, list]]:
    obs_dim = observation_space_size
    action_dim = ActionMap.n_simple()

    actor = ActorNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layer_sizes=[64, 64],
        hidden_layer_activation=nn.ReLU,
        opponent_action_dim=action_dim if CONSIDER_OPPONENT_ACTION else None,
    )

    critic_network = QNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layer_sizes=[128, 128],
        hidden_layer_activation=nn.ReLU,
        # Will setup Tanh with appropriate range
        is_footsies=False,
        use_dense_reward=True,
        opponent_action_dim=action_dim if CONSIDER_OPPONENT_ACTION else None,
    )

    target_network = deepcopy(critic_network)

    critic = QFunctionNetwork(
        q_network=critic_network,
        action_dim=action_dim,
        opponent_action_dim=action_dim,
        discount=0.99,
        learning_rate=1e-3,
        target_network=target_network,
        target_network_blank=True, # NOTE: only for the beginning of training!
        target_network_update_interval=1000,
    )

    learner = A2CQLearner(
        actor=actor,
        critic=critic,
        actor_learning_rate=1e-4,
        actor_entropy_loss_coef=0.1,
        policy_cumulative_discount=False,
        update_style="expected-sarsa",
    )

    a2c = A2CAgent(
        learner=learner,
        action_space_size=action_dim,
        footsies=True,
        use_opponents_perspective=False,
    )

    agent = TheOneAgent(
        observation_space_size=observation_space_size,
        action_space_size=action_space_size,
        representation=None,
        a2c=a2c,
        opponent_model=None,
        game_model=None,
        reaction_time_emulator=None,
        over_simple_actions=True,
        opponent_model_frameskip=True,
        game_model_learning_rate=1e-4,
        opponent_model_learning_rate=1e-4,
    )

    loggables = get_loggables(agent)

    return agent, loggables