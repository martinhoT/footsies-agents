from torch import nn
from agents.action import ActionMap
from agents.a2c.agent import A2CAgent
from agents.a2c.loggables import get_loggables
from agents.a2c.a2c import A2CQLearner, ActorNetwork
from agents.ql.ql import QFunctionTable


CONSIDER_OPPONENT_ACTION = True

def model_init(observation_space_size: int, action_space_size: int) -> tuple[A2CAgent, dict[str, list]]:
    obs_dim = observation_space_size
    action_dim = ActionMap.n_simple()

    actor = ActorNetwork(
        obs_dim=obs_dim + (action_dim if CONSIDER_OPPONENT_ACTION else 0),
        action_dim=action_dim,
        hidden_layer_sizes=[64, 64],
        hidden_layer_activation=nn.ReLU,
    )

    critic = QFunctionTable(
        action_dim=action_dim,
        opponent_action_dim=action_dim,
        discount=1.0,
        learning_rate=5e-1,
        table_as_matrix=False,
        environment="footsies",
    )

    learner = A2CQLearner(
        actor=actor,
        critic=critic,
        actor_learning_rate=1e-4,
        actor_entropy_loss_coef=0.1,
        policy_cumulative_discount=False,
        consider_opponent_action=CONSIDER_OPPONENT_ACTION,
    )

    agent = A2CAgent(
        learner=learner,
        action_space_size=action_dim,
    )

    loggables = get_loggables(agent)

    return agent, loggables