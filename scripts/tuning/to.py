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
    opponent_model_lr: float,
    actor_entropy_coef: float,
    opponent_model_entropy_coef: float,
    critic_hs: str = "[128, 128]",
    critic_activation: str = "nn.LeakyReLU",
    actor_hs: str = "[64, 64]",
    actor_activation: str = "nn.LeakyReLU",
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

        actor_lr=actor_lr,
        critic_lr=critic_lr,
        opponent_model_lr=opponent_model_lr,
        actor_entropy_coef=actor_entropy_coef,
        opponent_model_entropy_coef=opponent_model_entropy_coef,
        actor_gradient_clipping=actor_gradient_clipping,
        critic_target_update_rate=critic_target_update_rate,

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


def create_arch_suggestions(trial: optuna.Trial, label: str, use: bool = True) -> tuple[str, str]:
    activation_fn = trial.suggest_categorical(f"{label}_activation", ["nn.ReLU", "nn.LeakyReLU", "nn.Tanh"])
    hs = trial.suggest_categorical(f"{label}_hs", [
        "[64, 64]",
        "[128, 128]",
        "[64, 64, 64]",
    ])

    return activation_fn, hs


def define_model(trial: optuna.Trial, env: Env) -> TrainingLoggerWrapper[TheOneAgent]:
    # critic_arch_activation, critic_arch_hs = create_arch_suggestions(trial, "critic")
    # actor_arch_activation, actor_arch_hs = create_arch_suggestions(trial, "actor")

    loggables = {}
    agent = create_model_from_parameters(env,
        actor_lr=trial.suggest_float("actor_lr", 1e-5, 1e-1),
        critic_lr=trial.suggest_float("critic_lr", 1e-5, 1e-1),
        opponent_model_lr=trial.suggest_float("opponent_model_lr", 1e-5, 1e-1),
        actor_entropy_coef=trial.suggest_float("actor_entropy_coef", 0.0, 1.0),
        opponent_model_entropy_coef=trial.suggest_float("opponent_model_entropy_coef", 0.0, 1.0),
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
