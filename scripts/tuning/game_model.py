import optuna
from gymnasium import Env
from agents.game_model.agent import GameModelAgent


def define_model(trial: optuna.Trial, env: Env) -> GameModelAgent:
    hidden_layer_sizes_specification = trial.suggest_categorical("hidden_layer_sizes", [
        "",
        "128",
        "128,128",
        "128,128,128",
    ])
    
    if hidden_layer_sizes_specification:
        hidden_layer_activation_specification = trial.suggest_categorical("hidden_layer_activation", [
            "ReLU",
            "LeakyReLU",
            "Sigmoid",
            "Sigmoid.Identity",
        ])
    
    else:
        hidden_layer_activation_specification = "Identity"

    return GameModelAgent(
        observation_space_size=env.observation_space.shape[0],
        action_space_size=env.action_space.n,
        by_primitive_actions=False,
        by_observation_differences=trial.suggest_categorical("by_observation_differences", [False, True]),
        move_transition_scale=trial.suggest_int("move_transition_scale", 1, 1000),
        mini_batch_size=trial.suggest_int("mini_batch_size", 1, 1000),
        learning_rate=trial.suggest_float("learning_rate", 0.001, 0.5),
        hidden_layer_sizes_specification=hidden_layer_sizes_specification,
        hidden_layer_activation_specification=hidden_layer_activation_specification,
    )


def objective(agent: GameModelAgent, env: Env) -> float:
    return agent.evaluate_average_loss_and_clear()
