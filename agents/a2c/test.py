import torch
import gymnasium
from torch import nn
from gymnasium.wrappers.flatten_observation import FlattenObservation
from agents.a2c.a2c import A2CModule, ActorNetwork, CriticNetwork
from itertools import count

ENVIRONMENT = "CartPole-v1"

if ENVIRONMENT == "FrozenLake-v1":
    kwargs = {
        "is_slippery": False
    }

else:
    kwargs = {}

env = FlattenObservation(
    gymnasium.make(
        ENVIRONMENT,
        **kwargs,
        render_mode=None,
    )
)

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = A2CModule(
    actor=ActorNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layer_sizes=[],
        hidden_layer_activation=nn.Identity,
    ),
    critic=CriticNetwork(
        obs_dim=obs_dim,
        hidden_layer_sizes=[],
        hidden_layer_activation=nn.Identity,
    ),
    discount=0.99,
    actor_learning_rate=1e-3,
    critic_learning_rate=1e-3,
)

try:
    terminated, truncated = True, True

    for i in count():
        while not (terminated or truncated):
            action = model.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            model.update(obs, next_obs, reward, terminated)
            obs = next_obs

        obs, info = env.reset()
        terminated, truncated = False, False

        if ENVIRONMENT == "FrozenLake-v1":
            print(model.value(torch.eye(16)).reshape(4, 4), "\x1B[4A", sep="")
        else:
            print(i, end="\r")

except KeyboardInterrupt:
    print("Value function:")
    if ENVIRONMENT == "FrozenLake-v1":
        print(model.value(torch.eye(16)).reshape(4, 4))
    else:
        print(None)

    print("Policy:")
    if ENVIRONMENT == "FrozenLake-v1":
        print(model.policy(torch.eye(16)).reshape(4, 4, 4))
    else:
        print(None)

env.close()

env = FlattenObservation(
    gymnasium.make(
        ENVIRONMENT,
        **kwargs,
        render_mode="human",
    )
)

terminated, truncated = True, True

while True:
    while not (terminated or truncated):
        action = model.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)

    obs, info = env.reset()
    terminated, truncated = False, False
