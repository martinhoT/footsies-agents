import optuna
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
from gymnasium import Env
from torch import nn


def create_model_from_parameters(env: Env, activation_fn: str, net_arch: str, learning_rate: float, tau: float, gamma: float) -> DQN:
    activation_fn = eval(activation_fn, {"nn": nn})
    net_arch = eval(net_arch)
    
    return DQN(
        # Base
        policy="MlpPolicy",
        env=env,
        
        # Hyperparameters
        learning_rate=learning_rate,
        tau=tau,
        policy_kwargs={
            "activation_fn": activation_fn,
            "net_arch": net_arch
        },
        gamma=gamma,

        # Always the same
        max_grad_norm=0.5,
    )


def define_model(trial: optuna.Trial, env: Env) -> DQN:
    return create_model_from_parameters(env,
        activation_fn=trial.suggest_categorical("activation_fn", ["nn.ReLU", "nn.LeakyReLU", "nn.Tanh"]),
        net_arch=trial.suggest_categorical("net_arch", [
            "[64, 64]",
            "[128, 128]",
            "[64, 64, 64]",
        ]),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        tau=trial.suggest_float("tau", 0.0, 1.0),
        gamma=trial.suggest_float("gamma", 0.0, 1.0),
    )


def objective(agent: BaseAlgorithm, env: Env) -> float:
    reward_mean, reward_std = evaluate_policy(agent, env,
        n_eval_episodes=100,
        deterministic=False,
    )

    assert isinstance(reward_mean, float)

    return reward_mean
