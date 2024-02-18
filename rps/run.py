import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Generator
from torch import nn
from tqdm import tqdm
from rps.rps import RPS
from rps.plotter import AnimatedPlot
from agents.a2c.a2c import A2CModule, ActorNetwork, CriticNetwork
from itertools import starmap, count


OPPONENT_POOL = {
    "uniform_random_play": lambda o, i: random.randint(0, 2),
    "rocky_play": lambda o, i: 0 if random.random() < 0.8 else random.randint(1, 2),
    "scissors_play": lambda o, i: 2 if random.random() < 0.9 else random.randint(0, 1),
    "imitator": lambda o, i: i["p1_action"] if i["step"] > 0 else random.randint(0, 2),
}


def print_results(game: RPS, agent: A2CModule):
    healths_list = [(h1 + 1, h2 + 1) for h1 in range(game.health) for h2 in range(game.health)]
    plays_list = [(pl1, pl2) for pl1 in range(game.play_dim) for pl2 in range(game.play_dim)] if game.observation_include_play else [[None, None]]

    observations = starmap(game.craft_observation, sorted({
        tuple((*healths, *plays))
        for healths in healths_list
        for plays in plays_list
    }))

    results = {
        observation: (
            agent.value(observation).item(),
            agent.policy(observation).squeeze(),
        )
        for observation in observations
    }

    print("Value function:")
    for obs, (value, _) in results.items():
        print(f"{obs}: {value:.2f}")

    print()

    print("Policy:")
    for obs, (_, distribution) in results.items():
        print(f"{obs}: {distribution}")


def train(game: RPS, agent: A2CModule, episodes: int = None) -> Generator[tuple, None, None]:
    episode_iterator = range(episodes) if episodes is not None else count()

    try:
        for episode in tqdm(episode_iterator):
            obs, info = game.reset()
            terminated, truncated = False, False
            episode_reward = 0

            while not (terminated or truncated):
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, info = game.step(action)
                agent.update(obs, next_obs, reward, terminated)
                obs = next_obs

                episode_reward += reward
            
            yield episode, agent.delta, episode_reward
    
    except KeyboardInterrupt:
        pass


def exponential_moving_average(x: tuple[float], factor: float) -> list[float]:
    if len(x) > 1000:
        res = list(x)
        prev = res[0]
        for i in tqdm(range(len(res))):
            res[i] = factor * prev + (1 - factor) * res[i]
            prev = res[i]
        
        return res
            
    # NOTE: if you have enough memory...
    else:
        x = np.array(x)
        c = np.tril(np.tile(x, (x.size, 1)))
        if factor == 0.0:
            e = np.ones_like(x)
        else:
            e = (1 - factor) * factor ** x
        return np.sum(e[::-1].reshape((-1, 1)) * c, axis=1)


def main(plot: bool = False):
    torch.set_printoptions(precision=2, sci_mode=False)

    game = RPS(
        opponent=OPPONENT_POOL["imitator"],
        dense_reward=False,
        health=3,
        flattened=True,
        observation_include_play=False,
        observation_transformer=lambda o: o.to(torch.float32).unsqueeze(0),
    )

    agent = A2CModule(
        actor=ActorNetwork(
            obs_dim=game.observation_dim,
            action_dim=game.action_dim,
            hidden_layer_sizes=[],
            hidden_layer_activation=nn.Identity,
        ),
        critic=CriticNetwork(
            obs_dim=game.observation_dim,
            hidden_layer_sizes=[],
            hidden_layer_activation=nn.Identity,
        ),
        actor_learning_rate=1e-2,
        critic_learning_rate=1e-2,
        actor_eligibility_traces_decay=0.0,
        critic_eligibility_traces_decay=0.0,
        optimizer=torch.optim.SGD,
    )

    training_loop = train(game, agent)

    if plot:
        animation = AnimatedPlot()
        animation.setup(training_loop)
        animation.start()
        
    else:
        data = list(training_loop)
        episodes, deltas, rewards = zip(*data)

        print("Smooting deltas")
        deltas = exponential_moving_average(deltas, 0.99)
        print("Smoothing rewards")
        rewards = exponential_moving_average(rewards, 0.99)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        ax1.plot(episodes, deltas)
        ax2.plot(episodes, rewards)
        plt.show()

    print_results(game, agent)


if __name__ == "__main__":
    main(
        plot=False
    )