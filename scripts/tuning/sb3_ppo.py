import optuna
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from gymnasium import Env
from torch import nn


def create_model_from_parameters(env: Env, activation_fn: str, pi_arch: str, vf_arch: str, learning_rate: float, gae_lambda: float, ent_coef: float, gamma: float) -> PPO:
    activation_fn = eval(activation_fn, {"nn": nn})
    pi_arch = eval(pi_arch)
    vf_arch = eval(vf_arch)
    
    return PPO(
        # Base
        policy="MlpPolicy",
        env=env,
        
        # Hyperparameters
        learning_rate=learning_rate,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        policy_kwargs={
            "activation_fn": activation_fn,
            "net_arch": {
                "pi": pi_arch,
                "vf": vf_arch,
            }
        },
        gamma=gamma,

        # Always the same
        normalize_advantage=True,
        max_grad_norm=0.5,
        use_sde=False,
    )


def define_model(trial: optuna.Trial, env: Env) -> PPO:
    return create_model_from_parameters(env,
        activation_fn=trial.suggest_categorical("activation_fn", ["nn.ReLU", "nn.LeakyReLU", "nn.Tanh"]),
        pi_arch=trial.suggest_categorical("pi_arch", [
            "[64, 64]",
            "[128, 128]",
            "[64, 64, 64]",
        ]),
        vf_arch=trial.suggest_categorical("vf_arch", [
            "[64, 64]",
            "[128, 128]",
            "[64, 64, 64]",
        ]),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        gae_lambda=trial.suggest_float("gae_lambda", 0.0, 1.0),
        ent_coef=trial.suggest_float("ent_coef", 0.0, 1.0),
        gamma=trial.suggest_float("gamma", 0.0, 1.0),
    )


def objective(agent: BaseAlgorithm, env: Env) -> float:
    reward_mean, reward_std = evaluate_policy(agent, env,
        n_eval_episodes=100,
        deterministic=True,
    )

    assert isinstance(reward_mean, float)

    return reward_mean
