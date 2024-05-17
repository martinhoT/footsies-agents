from tyro import conf
from torch import nn
from copy import deepcopy
from typing import Literal
from agents.action import ActionMap
from agents.the_one.agent import TheOneAgent
from agents.a2c.agent import A2CAgent
from agents.a2c.a2c import A2CQLearner, ActorNetwork
from agents.ql.ql import QFunctionNetwork, QNetwork, QFunctionTable
from agents.the_one.loggables import get_loggables
from agents.mimic.mimic import PlayerModel, PlayerModelNetwork, ScarStore
from agents.mimic.agent import MimicAgent
from agents.the_one.reaction_time import ReactionTimeEmulator
from agents.game_model.agent import GameModelAgent
from agents.game_model.game_model import GameModel, GameModelNetwork



REACTION_TIME_MIN = 15
REACTION_TIME_MAX = 29

def model_init(observation_space_size: int, action_space_size: int, *,
    # Important modifiers
    use_reaction_time: bool = False,
    use_game_model: bool = True,
    game_model_skippers: bool = True,
    game_model_method: Literal["residual", "differences", "normal"] = "differences",
    
    # Learn what?
    learn: Literal["gm", "no-gm", "none", "all"] = "no-gm",

    # Opponent modifiers
    rollback: bool = False,
    use_opponent_model: bool = True,
    critic_opponent_update: str = "expected_sarsa",
    consider_explicit_opponent_policy: bool = True,
    opponent_model_dynamic_loss_weights: bool = False,
    opponent_model_recurrent: bool = True,
    opponent_model_reset_context_at: Literal["hit", "neutral", "end"] = "end",
    
    # Highly-tunable hyperparameters
    actor_lr: float = 9e-2,
    critic_lr: float = 5e-2,
    actor_entropy_coef: float = 0.036,
    actor_gradient_clipping: float = 0.5,
    critic_discount: float = 1.0,
    opponent_model_entropy_coef: float = 0.42,
    opponent_model_lr: float = 4e-2,

    # Miscellaneous, but should be scrutinized
    accumulate_at_frameskip: bool = True,
    alternative_advantage: bool = True,
    reaction_time_constant: bool = False,
    action_masking: bool = False,
    game_model_skippers_every: int = 5,
    game_model_single_skipper: int = 1,
    consider_opponent_at_all: bool = True,
    one_decision_at_hitstop: bool = True,

    # Probably should be kept as-is
    ppo: bool = False,
    perceive_intrinsic_reward: bool = False,
    policy_cumulative_discount: bool = False,
    maxent: float = 0.0,
    maxent_gradient_flow: bool = False,
    critic_tanh: bool = False,
    critic_agent_update: str = "expected_sarsa",
    critic_target_update_rate: int = 1000,
    critic_table: bool = False,
    act_with_qvalues: bool = False,

    # Can only be specified programatically
    critic_arch_hs: conf.Suppress[list[int]] = [128, 128],
    critic_arch_activation: conf.Suppress[type[nn.Module]] = nn.LeakyReLU,
    actor_arch_hs: conf.Suppress[list[int]] = [64, 64],
    actor_arch_activation: conf.Suppress[type[nn.Module]] = nn.LeakyReLU,
    opponent_model_arch_hs: conf.Suppress[list[int]] = [64, 64],
    opponent_model_arch_activation: conf.Suppress[type[nn.Module]] = nn.LeakyReLU,
    game_model_arch_hs: conf.Suppress[list[int]] = [64, 64],
    game_model_arch_activation: conf.Suppress[type[nn.Module]] = nn.LeakyReLU,
) -> tuple[TheOneAgent, dict[str, list]]:

    # Disable the opponent model when
    if not consider_opponent_at_all and use_opponent_model:
        raise ValueError("it does not make sense to not consider the opponent and use an opponent model")

    obs_dim = observation_space_size
    action_dim = action_space_size
    opponent_action_dim = ActionMap.n_simple() if consider_opponent_at_all else 1

    actor = ActorNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layer_sizes=actor_arch_hs,
        hidden_layer_activation=actor_arch_activation,
        opponent_action_dim=opponent_action_dim,
        footsies_masking=action_masking,
        p1=True,
    )

    if critic_table:
        critic = QFunctionTable(
            action_dim=action_dim,
            opponent_action_dim=opponent_action_dim,
            discount=critic_discount,
            learning_rate=critic_lr,
            table_as_matrix=False,
            environment="footsies",
        )
    
    else:
        critic_network = QNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_layer_sizes=critic_arch_hs,
            hidden_layer_activation=critic_arch_activation,
            # Will setup Tanh with appropriate range
            is_footsies=critic_tanh,
            use_dense_reward=False,
            opponent_action_dim=opponent_action_dim
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
        policy_cumulative_discount=policy_cumulative_discount,
        actor_gradient_clipping=actor_gradient_clipping,
        agent_update_style=getattr(A2CQLearner.UpdateStyle, critic_agent_update.upper()),
        opponent_update_style=getattr(A2CQLearner.UpdateStyle, critic_opponent_update.upper()),
        maxent=maxent,
        maxent_gradient_flow=maxent_gradient_flow,
        ppo_objective=ppo,
        alternative_advantage=alternative_advantage,
        accumulate_at_frameskip=accumulate_at_frameskip,
        intrinsic_critic=intrinsic_critic,
    )

    a2c = A2CAgent(
        learner=learner,
        opponent_action_dim=opponent_action_dim,
        consider_explicit_opponent_policy=consider_explicit_opponent_policy,
        act_with_qvalues=act_with_qvalues,
        one_decision_at_hitstop=one_decision_at_hitstop,
    )

    if use_opponent_model:
        player_model = PlayerModel(
            player_model_network=PlayerModelNetwork(
                obs_dim=obs_dim,
                action_dim=opponent_action_dim,
                use_sigmoid_output=False,
                input_clip=False,
                input_clip_leaky_coef=0.02,
                hidden_layer_sizes=opponent_model_arch_hs,
                hidden_layer_activation=opponent_model_arch_activation,
                recurrent=opponent_model_recurrent,
            ),
            scar_store=ScarStore(
                obs_dim=obs_dim,
                max_size=1,
                min_loss=float("+inf"),
            ),
            learning_rate=opponent_model_lr,
            loss_dynamic_weights=opponent_model_dynamic_loss_weights,
            loss_dynamic_weights_max=10.0,
            entropy_coef=opponent_model_entropy_coef,
            reset_context_at=opponent_model_reset_context_at,
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
        assert opponent_model is not None
        assert opponent_model.p2_model is not None
        
        reaction_time_emulator = ReactionTimeEmulator(
            actor=actor,
            opponent=opponent_model.p2_model,
            history_size=30,
            # These don't matter since they will be substituted by the call below
            multiplier=1.0,
            additive=0.0,
        )
        reaction_time_emulator.confine_to_range(REACTION_TIME_MIN, REACTION_TIME_MAX, action_dim)

        if reaction_time_constant:
            reaction_time_emulator.constant = True

    else:
        reaction_time_emulator = None

    if use_game_model:
        residual = False
        by_differences = False
        if game_model_method == "differences":
            by_differences = True
        elif game_model_method == "normal":
            pass
        elif game_model_method == "residual":
            residual = True

        game_model = GameModel(
            game_model_network=GameModelNetwork(
                obs_dim=obs_dim,
                p1_action_dim=action_dim,
                p2_action_dim=opponent_action_dim,
                hidden_layer_sizes=game_model_arch_hs,
                hidden_layer_activation=game_model_arch_activation,
                residual=residual,
            ),
            learning_rate=1e-2,
            discrete_conversion=False,
            discrete_guard=False,
            by_differences=by_differences,
        )

        steps_n = list(range(REACTION_TIME_MIN, REACTION_TIME_MAX + 1, game_model_skippers_every)) if game_model_skippers else [game_model_single_skipper]

        game_model_agent = GameModelAgent(
            game_model,
            steps_n=steps_n
        )

    else:
        game_model_agent = None

    if learn == "all":
        learn_a2c, learn_game_model, learn_opponent_model = True, True, True
    elif learn == "none":
        learn_a2c, learn_game_model, learn_opponent_model = False, False, False
    elif learn == "gm":
        learn_a2c, learn_game_model, learn_opponent_model = False, True, False
    elif learn == "no-gm":
        learn_a2c, learn_game_model, learn_opponent_model = True, False, True
    else:
        raise ValueError(f"wrong value for 'learn' ('{learn}'), should be one of ('all', 'none', 'gm', 'no-gm')")

    agent = TheOneAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        opponent_action_dim=opponent_action_dim,
        a2c=a2c,
        opponent_model=opponent_model,
        game_model=game_model_agent,
        reaction_time_emulator=reaction_time_emulator,
        rollback_as_opponent_model=rollback,
        # Learn?
        learn_a2c=learn_a2c,
        learn_game_model=learn_game_model,
        learn_opponent_model=learn_opponent_model,
    )

    loggables = get_loggables(agent)

    return agent, loggables
