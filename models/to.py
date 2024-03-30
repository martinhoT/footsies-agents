from torch import nn
from copy import deepcopy
from agents.action import ActionMap
from agents.the_one.agent import FootsiesAgent as TheOneAgent
from agents.a2c.agent import FootsiesAgent as A2CAgent
from agents.a2c.a2c import A2CQLearner, ActorNetwork
from agents.ql.ql import QFunctionNetwork, QNetwork, QFunctionTable
from agents.the_one.loggables import get_loggables
from agents.mimic.mimic import PlayerModel, PlayerModelNetwork, ScarStore
from agents.mimic.agent import FootsiesAgent as MimicAgent
from agents.the_one.reaction_time import ReactionTimeEmulator


CONSIDER_OPPONENT_ACTION = True

def model_init(observation_space_size: int, action_space_size: int, *,
    # Important modifiers
    ppo: bool = False,
    remove_special_moves: bool = True,
    perceive_intrinsic_reward: bool = False,
    use_reaction_time: bool = False,
    maxent: float = 0.0,
    maxent_gradient_flow: bool = False,
    
    # Opponent modifiers
    rollback: bool = False,
    use_opponent_model: bool = False,
    critic_opponent_update: str = "expected_sarsa",
    consider_explicit_opponent_policy: bool = True,
    opponent_model_dynamic_loss_weights: bool = True,
    
    # Highly-tunable hyperparameters
    actor_lr: float = 3e-2,
    critic_lr: float = 1e-2,
    actor_entropy_coef: float = 0.04,
    actor_gradient_clipping: float = 0.5,
    critic_discount: float = 0.9,

    # Miscellaneous, but should be scrutinized
    broadcast_at_frameskip: bool = False,
    accumulate_at_frameskip: bool = True,
    alternative_advantage: bool = True,

    # Probably should be kept as-is
    critic_tanh: bool = False,
    critic_agent_update: str = "expected_sarsa",
    critic_target_update_rate: int = 1000,
    critic_table: bool = False,
    act_with_qvalues: bool = False,
    action_masking: bool = True,
) -> tuple[TheOneAgent, dict[str, list]]:

    obs_dim = observation_space_size
    action_dim = ActionMap.n_simple() - (2 if remove_special_moves else 0)
    opponent_action_dim = ActionMap.n_simple()

    actor = ActorNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layer_sizes=[64, 64],
        hidden_layer_activation=nn.LeakyReLU,
        opponent_action_dim=opponent_action_dim if CONSIDER_OPPONENT_ACTION else None,
        footsies_masking=action_masking,
        p1=True,
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
        maxent=maxent,
        maxent_gradient_flow=maxent_gradient_flow,
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

    if use_opponent_model:
        player_model = PlayerModel(
            player_model_network=PlayerModelNetwork(
                obs_dim=obs_dim,
                action_dim=opponent_action_dim,
                use_sigmoid_output=False,
                input_clip=False,
                input_clip_leaky_coef=0.02,
                hidden_layer_sizes=[64, 64],
                hidden_layer_activation=nn.LeakyReLU,
            ),
            scar_store=ScarStore(
                obs_dim=obs_dim,
                max_size=1,
                min_loss=float("+inf"),
            ),
            learning_rate=1e-2,
            loss_dynamic_weights=opponent_model_dynamic_loss_weights,
            loss_dynamic_weights_max=10.0,
            entropy_coef=0.3,
        )

        opponent_model = MimicAgent(
            action_dim=opponent_action_dim,
            by_primitive_actions=False,
            p1_model=None,
            p2_model=player_model,
        )

    else:
        opponent_model = None

    if use_reaction_time:
        reaction_time_emulator = ReactionTimeEmulator(
            inaction_probability=0.0,
            history_size=30,
            # These don't matter since they will be substituted by the call below
            multiplier=1.0,
            additive=0.0,
        )
        reaction_time_emulator.confine_to_range(15, 29, action_dim)

    else:
        reaction_time_emulator = None

    agent = TheOneAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        opponent_action_dim=opponent_action_dim if CONSIDER_OPPONENT_ACTION else None,
        a2c=a2c,
        opponent_model=opponent_model,
        reaction_time_emulator=reaction_time_emulator,
        remove_special_moves=remove_special_moves,
        rollback_as_opponent_model=rollback,
        # Not used
        representation=None,
        game_model=None,
        game_model_learning_rate=1e-4,
    )

    loggables = get_loggables(agent)

    return agent, loggables
