from collections import deque
import numpy as np
import torch
import gymnasium
import matplotlib.pyplot as plt
from typing import Any
from torch import nn
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.normalize import NormalizeObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from tqdm import tqdm
from agents.a2c.a2c import A2CLambdaLearner, ActorNetwork, CriticNetwork
from agents.icm import IntrinsicCuriosityModule, AbstractEnvironmentEncoder, InverseEnvironmentModel, ForwardEnvironmentModel, NoveltyTable
from agents.tile import TileCoding, Tiling
from itertools import combinations, count
from enum import Enum

class CartPoleAttribute(Enum):
    CART_POSITION = 0
    CART_VELOCITY = 1
    POLE_ANGLE = 2
    POLE_ANGULAR_VELOCITY = 3


class MountainCarAttribute(Enum):
    POSITION = 0
    VELOCITY = 1


class CartPoleCoding(ObservationWrapper):
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


class MountainCarCoding(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        single_tilings = [
            Tiling({
                MountainCarAttribute.POSITION: np.linspace(-1.2, 0.6, 20)
            }),
            Tiling({
                MountainCarAttribute.VELOCITY: np.linspace(-0.07, 0.07, 20)
            }),
        ]

        self.coding = TileCoding([
            *single_tilings,
            *(t + 0.05 for t in single_tilings),
            *(t - 0.05 for t in single_tilings),
            Tiling({
                MountainCarAttribute.POSITION: np.linspace(-1.2, 0.6, 20),
                MountainCarAttribute.VELOCITY: np.linspace(-0.07, 0.07, 20),
            })
        ])

        self.observation_space = Box(low=0.0, high=1.0, shape=(self.coding.size,))

        print("Using coding with size ", self.coding.size)

    def observation(self, observation: np.ndarray) -> torch.Tensor:
        return 1.0 * self.coding.transform(observation)


"""Good sets of parameters
FrozenLake:
    Actor LR: 1.0e-1
    Critic LR: 2.0e-2
    Linear
CartPole:
    Actor LR: 1.0e-4
    Critic LR: 1.0e-3
    Traces: 0.8 for both
    1 hidden layer of size 128 with ReLU activations
    Adam optimizer
MountainCar: (not good actually)
    Actor LR: 0.475
    Critic LR: 0.469
    Actor ET: 0.41
    Critic ET: 0.89
    Discount: 0.295
    1 hidden layer of size (128, 256) with ReLU activations (actor, critic)
"""


ENVIRONMENT = "MountainCar-v0"

if ENVIRONMENT == "FrozenLake-v1":
    kwargs = {
        "is_slippery": False
    }

elif ENVIRONMENT == "LunarLander-v2":
    kwargs = {
        "continuous": False,
        "enable_wind": True,
    }

else:
    kwargs = {}

env_generator = lambda e: (
    # MountainCarCoding(
    # CartPoleCoding(
    TransformObservation(
        FlattenObservation(
            e
        )
        , lambda obs: torch.from_numpy((obs - np.array([-0.3, 0.0])) / np.array([0.9, 0.07])).float().unsqueeze(0)
    )
    # )
)

env = env_generator(
    gymnasium.make(
        ENVIRONMENT,
        **kwargs,
        render_mode="human",
    )
)

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

learner = A2CLambdaLearner(
    actor=ActorNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layer_sizes=[32],
        hidden_layer_activation=nn.ReLU,
    ),
    critic=CriticNetwork(
        obs_dim=obs_dim,
        hidden_layer_sizes=[32],
        hidden_layer_activation=nn.ReLU,
    ),
    discount=1.0,
    actor_lambda=0.8,
    critic_lambda=0.8,
    actor_entropy_loss_coef=0.0,
    actor_optimizer=torch.optim.Adam,
    critic_optimizer=torch.optim.Adam,
    **{
        "actor_optimizer.lr": 1e-3,
        "critic_optimizer.lr": 1e-3,
    }
)

# Novelty-based intrinsic reward
novelty_table = NoveltyTable(reward_scale=10)
single_tilings = [
    Tiling({
        MountainCarAttribute.POSITION: torch.linspace(-1.0, 1.0, 20)
    }),
    Tiling({
        MountainCarAttribute.VELOCITY: torch.linspace(-1.0, 1.0, 20)
    }),
]
mountain_car_tile_coding = TileCoding([
    *single_tilings,
    *(t + 0.05 for t in single_tilings),
    *(t - 0.05 for t in single_tilings),
    Tiling({
        MountainCarAttribute.POSITION: torch.linspace(-1.0, 1.0, 20),
        MountainCarAttribute.VELOCITY: torch.linspace(-1.0, 1.0, 20),
    })
])

# Curiosity-based intrinsic reward
curiosity_trainer = None

# Value function visualization
if ENVIRONMENT == "MountainCar-v0":
    from plot_utils import Heatmap
    # x = torch.linspace(-1.2, 0.6, 20)
    # y = torch.linspace(-0.07, 0.07, 20)
    x = torch.linspace(-1.0, 1.0, 20)
    y = torch.linspace(-1.0, 1.0, 20)
    heatmap_grid = torch.meshgrid(x, y, indexing="xy")
    heatmap_states = torch.stack(heatmap_grid, dim=-1)
    heatmap = Heatmap(
        learner.critic(heatmap_states).detach().numpy().squeeze(),
        xlabel="Position",
        ylabel="Velocity",
        xticks=[f"{v.item():.2f}" for v in x],
        yticks=[f"{v.item():.2f}" for v in y],
        title="Value function",
    )
    heatmap.plot()
    
    ans = input("Does heatmap look good?")
    if ans == "n":
        exit(0)

else:
    heatmap = None

try:
    terminated, truncated = True, True

    # episode_iterator = count()
    episode_iterator = range(10000)
    step = 0
    scores = []
    scores_avg = []
    recent_scores = deque([], maxlen=100)
    deltas = []
    inv_losses = []
    fwd_losses = []
    inv_loss_exp_avg = 0.0
    fwd_loss_exp_avg = 0.0
    heatmap_update_interval = 25
    for i in tqdm(episode_iterator):
    # for i in count():
        score = 0
        while not (terminated or truncated):
            action = learner.sample_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # Augment reward with novelty-based curiosity
            t = mountain_car_tile_coding.transform(next_obs)
            novelty_table.register(t)
            reward += novelty_table.intrinsic_reward(t)
            # Update agent
            learner.learn(obs, next_obs, reward, terminated)
            
            obs = next_obs
            step += 1
            score += reward
            deltas.append(learner.delta)
            if curiosity_trainer is not None:
                inv_loss_exp_avg = 0.99 * inv_loss_exp_avg + 0.01 * curiosity_trainer.inverse_model_loss
                fwd_loss_exp_avg = 0.99 * fwd_loss_exp_avg + 0.01 * curiosity_trainer.forward_model_loss

            if ENVIRONMENT == "MountainCar-v0" and terminated:
                print("VICTORY!!!")

            if heatmap and step % heatmap_update_interval == 0:
                heatmap.update(learner.critic(heatmap_states).detach().numpy().squeeze())

        obs, info = env.reset()
        terminated, truncated = False, False
        scores.append(score)
        recent_scores.append(score)
        scores_avg.append(sum(recent_scores) / len(recent_scores))
        if curiosity_trainer is not None:
            inv_losses.append(inv_loss_exp_avg)
            fwd_losses.append(fwd_loss_exp_avg)

        if ENVIRONMENT == "FrozenLake-v1":
            print(learner.value(torch.eye(16)).reshape(4, 4), "\x1B[4A", sep="")

except KeyboardInterrupt:
    pass

print("Value function:")
if ENVIRONMENT == "FrozenLake-v1":
    print(learner.value(torch.eye(16)).reshape(4, 4))
else:
    print(None)

print("Policy:")
if ENVIRONMENT == "FrozenLake-v1":
    print(learner.policy(torch.eye(16)).reshape(4, 4, 4))
else:
    print(None)

plt.plot(deltas)
plt.savefig("a2c_test_deltas")
plt.clf()

plt.plot(scores)
plt.plot(scores_avg)
plt.savefig("a2c_test_scores")
plt.clf()

if curiosity_trainer is not None:
    plt.plot(inv_losses)
    plt.savefig("a2c_test_curio_inv_losses")
    plt.clf()

    plt.plot(fwd_losses)
    plt.savefig("a2c_test_curio_fwd_losses")
    plt.clf()

env.close()

env = env_generator(
    gymnasium.make(
        ENVIRONMENT,
        **kwargs,
        render_mode="human",
    )
)

terminated, truncated = True, True

while True:
    while not (terminated or truncated):
        action = learner.sample_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        # Augment reward with novelty-based curiosity
        # t = mountain_car_tile_coding.transform(next_obs)
        # novelty_table.register(t)
        # reward += novelty_table.intrinsic_reward(t)
        # Update agent
        # learner.learn(obs, next_obs, reward, terminated)
        obs = next_obs
        # print(learner.critic(learner._obs_to_torch(next_obs)).item())

    obs, info = env.reset()
    terminated, truncated = False, False
