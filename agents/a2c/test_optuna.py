from collections import deque
import numpy as np
import torch
import gymnasium
import matplotlib.pyplot as plt
import optuna
from typing import Any
from torch import nn
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from tqdm import tqdm
from agents.a2c.a2c import A2CModule, ActorNetwork, CriticNetwork
from agents.a2c.icm import IntrinsicCuriosityModule, AbstractEnvironmentEncoder, InverseEnvironmentModel, ForwardEnvironmentModel
from agents.tile import TileCoding, Tiling
from itertools import combinations, count
from enum import Enum

class CartPoleAttribute(Enum):
    CART_POSITION = 0
    CART_VELOCITY = 1
    POLE_ANGLE = 2
    POLE_ANGULAR_VELOCITY = 3


class CartPolePairs(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        single_tilings = [
            Tiling({
                CartPoleAttribute.CART_POSITION: np.linspace(-4.8, 4.8, 9)
            }),
            Tiling({
                CartPoleAttribute.CART_VELOCITY: np.linspace(-3.0, 3.0, 9)
            }),
            Tiling({
                CartPoleAttribute.POLE_ANGLE: np.linspace(-0.418, 0.418, 9)
            }),
            Tiling({
                CartPoleAttribute.POLE_ANGULAR_VELOCITY: np.linspace(-2.0, 2.0, 9)
            }),
        ]

        pair_tilings = [
            Tiling({
                **tiling1.breakpoints,
                **tiling2.breakpoints,
            }) for tiling1, tiling2 in combinations(single_tilings, 2)
        ]

        self.coding = TileCoding([
            # Singles
            *single_tilings,
            *(tiling + 0.5 for tiling in single_tilings),
            *(tiling - 0.5 for tiling in single_tilings),

            # Pairs
            *pair_tilings,
            *(tiling + 0.5 for tiling in pair_tilings),
            *(tiling - 0.5 for tiling in pair_tilings),

            # Triples
            # *(
            #     Tiling({
            #         **tiling1.breakpoints,
            #         **tiling2.breakpoints,
            #         **tiling3.breakpoints,
            #     }) for tiling1, tiling2, tiling3 in combinations(single_tilings, 3)
            # ),

            # All
            # Tiling({
            #     **single_tilings[0].breakpoints,
            #     **single_tilings[1].breakpoints,
            #     **single_tilings[2].breakpoints,
            #     **single_tilings[3].breakpoints,
            # })
        ])

        self.observation_space = Box(low=0.0, high=1.0, shape=(self.coding.size,))

    def observation(self, observation: np.ndarray) -> torch.Tensor:
        return 1.0 * self.coding.transform(observation)


ENVIRONMENT = "MountainCar-v0"

if ENVIRONMENT == "FrozenLake-v1":
    kwargs = {
        "is_slippery": False
    }

elif ENVIRONMENT == "LunarLander-v2":
    kwargs = {
        "continuous": False,
        "enable_wind": False,
    }

else:
    kwargs = {}

env_generator = lambda e: (
    # CartPolePairs(
        FlattenObservation(
            e
        )
    # )
)

env = env_generator(
    gymnasium.make(
        ENVIRONMENT,
        **kwargs,
        render_mode=None,
    )
)

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

def define_model(trial: optuna.Trial) -> A2CModule:
    encoded_dim = trial.suggest_int("encoded_dim", 4, 32)

    icm_encoder = AbstractEnvironmentEncoder(
        obs_dim=obs_dim,
        encoded_dim=encoded_dim,
        hidden_layer_sizes=[2**trial.suggest_int("encoder_hidden_layer_size", 1, 5)],
        hidden_layer_activation=nn.ReLU,
    )

    return A2CModule(
        actor=ActorNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_layer_sizes=[4],
            hidden_layer_activation=nn.ReLU,
        ),
        critic=CriticNetwork(
            obs_dim=obs_dim,
            hidden_layer_sizes=[4],
            hidden_layer_activation=nn.ReLU,
        ),
        discount=0.99,
        actor_learning_rate=trial.suggest_float("actor_lr", 1e-5, 5e-1),
        critic_learning_rate=trial.suggest_float("critic_lr", 1e-5, 5e-1),
        actor_eligibility_traces_decay=0.8,
        critic_eligibility_traces_decay=0.8,
        optimizer=torch.optim.Adam,
            curiosity=IntrinsicCuriosityModule(
            encoder=icm_encoder,
            inverse_model=InverseEnvironmentModel(
                encoded_dim=encoded_dim,
                action_dim=action_dim,
                encoder=icm_encoder,
                hidden_layer_sizes=[2**trial.suggest_int("inverse_model_hidden_layer_size", 1, 5)],
                hidden_layer_activation=nn.ReLU,
            ),
            forward_model=ForwardEnvironmentModel(
                encoded_dim=encoded_dim,
                action_dim=action_dim,
                encoder=icm_encoder,
                hidden_layer_sizes=[2**trial.suggest_int("forward_model_hidden_layer_size", 1, 5)],
                hidden_layer_activation=nn.ReLU,
            ),
            reward_scale=1e4,
        ),
    )

def optimize(trial: optuna.Trial, n_episodes: int = 1000) -> float:
    model = define_model(trial)
    
    episode_iterator = range(n_episodes)
    total_score = 0
    for _ in tqdm(episode_iterator):
        obs, _ = env.reset()
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = model.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            model.update(obs, next_obs, reward, terminated)
            
            obs = next_obs
            # Use intrinsic reward as well, or else it's too sparse for Optuna I'd guess
            total_score += reward + model.intrinsic_reward

    return total_score / n_episodes


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///a2c_icm_mountain_car.db",
        sampler=None,
        pruner=None,
        study_name="icm_mountain_car",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(optimize, n_trials=100, show_progress_bar=True)

    env.close()

    # Copied straight from https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))