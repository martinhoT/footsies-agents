import optuna
from agents.the_one.agent import TheOneAgent
from models import to_
from stable_baselines3 import PPO
from gymnasium import Env
from torch import nn
from typing import cast
from gymnasium.spaces import Discrete
from agents.logger import TrainingLoggerWrapper

def create_arch_suggestions(trial: optuna.Trial, label: str) -> tuple[type[nn.Module], list[int]]:
    activation_fn = eval(trial.suggest_categorical(f"{label}_activation", ["nn.ReLU", "nn.LeakyReLU", "nn.Tanh"]))
    hs = eval(trial.suggest_categorical(f"{label}_hs", [
        "[64, 64]",
        "[128, 128]",
        "[64, 64, 64]",
    ]))

    return activation_fn, hs

def define_model(trial: optuna.Trial, env: Env) -> TrainingLoggerWrapper[TheOneAgent]:

    assert env.observation_space.shape is not None
    assert isinstance(env.action_space, Discrete)

    critic_arch_activation, critic_arch_hs = create_arch_suggestions(trial, "critic")
    actor_arch_activation, actor_arch_hs = create_arch_suggestions(trial, "actor")
    opponent_model_arch_activation, opponent_model_arch_hs = create_arch_suggestions(trial, "opponent_model")
    # game_model_arch_activation, game_model_arch_hs = create_arch_suggestions(trial, "game_model")
    game_model_arch_activation, game_model_arch_hs = nn.LeakyReLU, [64, 64]

    actor_lr = trial.suggest_float("actor_lr", 1e-5, 1e-2, log=True)
    critic_lr = trial.suggest_float("critic_lr", 1e-5, 1e-2, log=True)

    actor_entropy_coef = trial.suggest_float("actor_entropy_coef", 0.0, 0.3)
    # actor_gradient_clipping = trial.suggest_float("actor_gradient_clipping", 0.0, 1.0)
    actor_gradient_clipping = 0.5

    # critic_target_update_rate = trial.suggest_int("critic_target_update_rate", 500, 10500, step=500)
    critic_target_update_rate = 1000

    agent, loggables = to_(
        observation_space_size=env.observation_space.shape[0],
        action_space_size=int(env.action_space.n),

        # Important modifiers
        remove_special_moves=True,
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
        critic_arch_hs=critic_arch_hs,
        critic_arch_activation=critic_arch_activation,
        actor_arch_hs=actor_arch_hs,
        actor_arch_activation=actor_arch_activation,
        opponent_model_arch_hs=opponent_model_arch_hs,
        opponent_model_arch_activation=opponent_model_arch_activation,
        game_model_arch_hs=game_model_arch_hs,
        game_model_arch_activation=game_model_arch_activation,
    )

    loggables["network_histograms"].clear()

    return TrainingLoggerWrapper[TheOneAgent](
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


def objective(agent: TrainingLoggerWrapper[TheOneAgent], env: Env) -> float:
    return  agent.win_rate
