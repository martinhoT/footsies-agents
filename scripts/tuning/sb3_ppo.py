import optuna
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from gymnasium import Env
from torch import nn


def create_model_from_parameters(env: Env, *,
    learning_rate: float,
    gae_lambda: float,
    ent_coef: float,
    vf_coef: float,
    gamma: float = 1.0,
    pi_arch: str = "[64, 64]",
    vf_arch: str = "[128, 128]",
    activation_fn: str = "nn.LeakyReLU",
) -> PPO:
    
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
        vf_coef=vf_coef,
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
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-1),
        gae_lambda=trial.suggest_float("gae_lambda", 0.0, 1.0),
        ent_coef=trial.suggest_float("ent_coef", 0.0, 1.0),
        vf_coef=trial.suggest_float("vf_coef", 0.0, 1.0),
    )


def objective(agent: BaseAlgorithm, env: Env) -> float:
    reward_mean, reward_std = evaluate_policy(agent, env,
        n_eval_episodes=100,
        deterministic=True,
    )

    assert isinstance(reward_mean, float)

    return reward_mean
