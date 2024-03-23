from torch import nn
from copy import deepcopy
from agents.action import ActionMap
from agents.the_one.agent import FootsiesAgent as TheOneAgent
from agents.a2c.agent import FootsiesAgent as A2CAgent
from agents.a2c.a2c import A2CQLearner, ActorNetwork
from agents.ql.ql import QFunctionNetwork, QNetwork, QFunctionTable
from agents.the_one.loggables import get_loggables


CONSIDER_OPPONENT_ACTION = True

def model_init(observation_space_size: int, action_space_size: int, *,
    actor_lr: float = 1e-2,
    critic_lr: float = 1e-2,
    actor_entropy_coef: float = 0.02,
    actor_gradient_clipping: float = 0.5,
    critic_tanh: bool = False,
    critic_discount: float = 0.9,
    critic_agent_update: str = "expected_sarsa",
    critic_opponent_update: str = "expected_sarsa",
    critic_target_update_rate: int = 1000,
    critic_table: bool = False,
    act_with_qvalues: bool = False,
    alternative_advantage: bool = True,
    broadcast_at_frameskip: bool = False,
    consider_explicit_opponent_policy: bool = False,
    accumulate_at_frameskip: bool = True,
    rollback: bool = False,
    perceive_intrinsic_reward: bool = False,
    ppo: bool = False,
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

    if critic_table:
        critic = QFunctionTable(
            action_dim=action_dim,
            opponent_action_dim=opponent_action_dim if CONSIDER_OPPONENT_ACTION else None,
            discount=critic_discount,
            learning_rate=critic_lr,
            table_as_matrix=False,
            environment="footsies",
        )
    
    else:
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
            target_network_update_interval=critic_target_update_rate,
        )

    intrinsic_critic = deepcopy(critic) if perceive_intrinsic_reward else None

    learner = A2CQLearner(
        actor=actor,
        critic=critic,
        actor_learning_rate=actor_lr,
        actor_entropy_loss_coef=actor_entropy_coef,
        policy_cumulative_discount=False,
        actor_gradient_clipping=actor_gradient_clipping,
        agent_update_style=getattr(A2CQLearner.UpdateStyle, critic_agent_update.upper()),
        opponent_update_style=getattr(A2CQLearner.UpdateStyle, critic_opponent_update.upper()),
        ppo_objective=ppo,
        alternative_advantage=alternative_advantage,
        accumulate_at_frameskip=accumulate_at_frameskip,
        broadcast_at_frameskip=broadcast_at_frameskip,
        intrinsic_critic=intrinsic_critic,
    )

    a2c = A2CAgent(
        learner=learner,
        opponent_action_dim=opponent_action_dim if CONSIDER_OPPONENT_ACTION else None,
        footsies=True,
        use_opponents_perspective=False,
        consider_explicit_opponent_policy=consider_explicit_opponent_policy,
        act_with_qvalues=act_with_qvalues,
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
        remove_special_moves=True,
        rollback_as_opponent_model=rollback,
        game_model_learning_rate=1e-4,
    )

    loggables = get_loggables(agent)

    return agent, loggables