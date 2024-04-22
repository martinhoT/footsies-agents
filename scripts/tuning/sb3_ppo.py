import optuna
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from gymnasium import Env
from torch import nn

def define_model(trial: optuna.Trial, env: Env) -> BaseAlgorithm:

    activation_fn = eval(trial.suggest_categorical("activation_fn", ["nn.ReLU", "nn.LeakyReLU", "nn.Tanh"]))
    pi_arch = eval(trial.suggest_categorical("pi_arch", [
        "[64, 64]",
        "[128, 128]",
        "[64, 64, 64]",
    ]))
    vf_arch = eval(trial.suggest_categorical("vf_arch", [
        "[64, 64]",
        "[128, 128]",
        "[64, 64, 64]",
    ]))

    return PPO(
        # Base
        policy="MlpPolicy",
        env=env,
        
        # Hyperparameters
        learning_rate=trial.suggest_float("learning_rate", 0.0, 1e-2),
        gae_lambda=trial.suggest_float("gae_lambda", 0.0, 1.0),
        ent_coef=trial.suggest_float("ent_coef", 0.0, 1.0),
        policy_kwargs={
            "activation_fn": activation_fn,
            "net_arch": {
                "pi": pi_arch,
                "vf": vf_arch,
            }
        },

        # Always the same
        gamma=1.0, # we already know that gamma=1.0 is the most successful, at least for our implementation
        normalize_advantage=True,
        max_grad_norm=0.5,
        use_sde=False,
    )


def objective(agent: BaseAlgorithm, env: Env) -> float:
    reward_mean, reward_std = evaluate_policy(agent, env,
        n_eval_episodes=100,
        deterministic=False,
    )

    return reward_mean # type: ignore
