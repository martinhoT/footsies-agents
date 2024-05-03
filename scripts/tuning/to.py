import optuna
from agents.the_one.agent import TheOneAgent
from models import to_
from stable_baselines3 import PPO
from gymnasium import Env
from torch import nn
from typing import cast
from gymnasium.spaces import Discrete
from agents.logger import TrainingLoggerWrapper


def create_model_from_parameters(env: Env,
    actor_lr: float,
    critic_lr: float,
    actor_entropy_coef: float,
    critic_hs: str,
    critic_activation: str,
    actor_hs: str,
    actor_activation: str,
    opponent_model_hs: str = "[64, 64]",
    opponent_model_activation: str = "nn.LeakyReLU",
    game_model_hs: str = "[64, 64]",
    game_model_activation: str = "nn.LeakyReLU",
    actor_gradient_clipping: float = 0.5,
    critic_target_update_rate: int = 1000,
    loggables_to_update: dict[str, list] | None = None
) -> TheOneAgent:

    assert env.observation_space.shape is not None
    assert isinstance(env.action_space, Discrete)

    critic_arch_hs_evaled = eval(critic_hs)
    critic_arch_activation_evaled = eval(critic_activation, {"nn": nn})
    actor_arch_hs_evaled = eval(actor_hs)
    actor_arch_activation_evaled = eval(actor_activation, {"nn": nn})
    opponent_model_arch_hs_evaled = eval(opponent_model_hs)
    opponent_model_arch_activation_evaled = eval(opponent_model_activation, {"nn": nn})
    game_model_arch_hs_evaled = eval(game_model_hs)
    game_model_arch_activation_evaled = eval(game_model_activation, {"nn": nn})

    agent, loggables = to_(
        observation_space_size=env.observation_space.shape[0],
        action_space_size=int(env.action_space.n),

        # Important modifiers
        use_reaction_time=False,
        use_game_model=True,
        game_model_method="differences",
        
        # Learn?
        learn="no-gm",

        # Opponent modifiers
        use_opponent_model=True,
        critic_opponent_update="expected_sarsa",
        consider_explicit_opponent_policy=True,
        opponent_model_dynamic_loss_weights=True,
        opponent_model_recurrent=True,
        opponent_model_reset_context_at="end",

        # Highly-tunable hyperparameters
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        actor_entropy_coef=actor_entropy_coef,
        actor_gradient_clipping=actor_gradient_clipping,
        critic_discount=1.0,

        # Miscellaneous, but should be scrutinized
        accumulate_at_frameskip=True,
        alternative_advantage=True,
        reaction_time_constant=False,
        action_masking=False,
        game_model_skippers_every=5,

        # Probably should be kept as-is
        ppo=False,
        perceive_intrinsic_reward=False,
        policy_cumulative_discount=False,
        maxent=0.0,
        maxent_gradient_flow=False,
        critic_tanh=False,
        critic_agent_update="expected_sarsa",
        critic_target_update_rate=critic_target_update_rate,
        critic_table=False,
        act_with_qvalues=False,

        # Can only be specified programatically
        critic_arch_hs=critic_arch_hs_evaled,
        critic_arch_activation=critic_arch_activation_evaled,
        actor_arch_hs=actor_arch_hs_evaled,
        actor_arch_activation=actor_arch_activation_evaled,
        opponent_model_arch_hs=opponent_model_arch_hs_evaled,
        opponent_model_arch_activation=opponent_model_arch_activation_evaled,
        game_model_arch_hs=game_model_arch_hs_evaled,
        game_model_arch_activation=game_model_arch_activation_evaled,
    )

    loggables["network_histograms"].clear()

    if loggables_to_update is not None:
        loggables_to_update.update(loggables)

    return agent


def create_arch_suggestions(trial: optuna.Trial, label: str) -> tuple[str, str]:
    activation_fn = trial.suggest_categorical(f"{label}_activation", ["nn.ReLU", "nn.LeakyReLU", "nn.Tanh"])
    hs = trial.suggest_categorical(f"{label}_hs", [
        "[64, 64]",
        "[128, 128]",
        "[64, 64, 64]",
    ])

    return activation_fn, hs


def define_model(trial: optuna.Trial, env: Env) -> TrainingLoggerWrapper[TheOneAgent]:
    critic_arch_activation, critic_arch_hs = create_arch_suggestions(trial, "critic")
    actor_arch_activation, actor_arch_hs = create_arch_suggestions(trial, "actor")

    loggables = {}
    agent = create_model_from_parameters(env,
        actor_lr=trial.suggest_float("actor_lr", 1e-5, 1e-1),
        critic_lr=trial.suggest_float("critic_lr", 1e-5, 1e-1),
        actor_entropy_coef=trial.suggest_float("actor_entropy_coef", 0.0, 0.3),
        critic_hs=critic_arch_hs,
        critic_activation=critic_arch_activation,
        actor_hs=actor_arch_hs,
        actor_activation=actor_arch_activation,
        loggables_to_update=loggables,
    )

    logged_agent = TrainingLoggerWrapper[TheOneAgent](
        agent,
        log_frequency=1000,
        log_dir=None,
        episode_reward=True,
        average_reward=True,
        average_reward_coef=None,
        win_rate=True,
        win_rate_over_last=100,
        truncation=True,
        episode_length=True,
        test_states_number=10000,
        step_start_value=0,
        episode_start_value=0,
        csv_save=False,
        **loggables,
    )

    return logged_agent
    

def objective(agent: TrainingLoggerWrapper[TheOneAgent], env: Env) -> float:
    return agent.win_rate
