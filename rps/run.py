import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import Generator
from torch import nn
from tqdm import tqdm
from rps.rps import RPS
from rps.plotter import AnimatedPlot
from agents.a2c.a2c import A2CLambdaLearner, ActorNetwork, CriticNetwork, A2CLearnerBase
from itertools import starmap, count


OPPONENT_POOL = {
    # RPS (no temporal actions)
    "uniform_random_play": lambda o, i: random.randint(0, 2),
    "rocky_play": lambda o, i: 0 if random.random() < 0.8 else random.randint(1, 2),
    "scissors_play": lambda o, i: 2 if random.random() < 0.9 else random.randint(0, 1),
    # FG (temporal actions)
    "uniform_random_play_temporal": lambda o, i: random.randint(0, 3),
    "masher": lambda o, i: ((0 if random.random() < 0.8 else random.randint(1, 3)) + 1) % 4,
    "thrower": lambda o, i: ((0 if random.random() < 0.9 else random.randint(1, 3)) + 2) % 4,
    "dodge-attacker": lambda o, i: 1 if i["p2_move"] == 3 else 3,
    # Special
    "imitator": lambda o, i: i["p1_action"] if i["step"] > 0 else random.randint(0, 2),
    "query": lambda o, i: int(input("Opponent action? ")),
}


def print_results(game: RPS, agent: A2CLambdaLearner):
    healths_list = [(h1 + 1, h2 + 1) for h1 in range(game.health) for h2 in range(game.health)]
    moves_list = [(pl1, pl2) for pl1 in range(game.move_dim) for pl2 in range(game.move_dim)] if game.observation_include_move else [[None, None]]

    observations = starmap(game.craft_observation, sorted({
        tuple((*healths, *moves))
        for healths in healths_list
        for moves in moves_list
    }))

    results = {
        observation: (
            agent.critic(observation).item(),
            agent.actor(observation).detach().squeeze(),
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
    print()


def train(game: RPS, agent: A2CLearnerBase, episodes: int = None, use_tqdm: bool = True) -> Generator[tuple, None, None]:
    episode_iterator = range(episodes) if episodes is not None else count()

    try:
        for episode in tqdm(episode_iterator, disable=not use_tqdm):
            obs, info = game.reset()
            terminated, truncated = False, False
            episode_reward = 0
            episode_length = 0

            while not (terminated or truncated):
                action = agent.sample_action(obs, info=info)
                next_obs, reward, terminated, truncated, info = game.step(action)
                # When the episode terminates, next_obs becomes None.
                # Make it equal to the previous observation so that the agent doesn't freak out, but it's not like it will actually use it.
                # next_obs = next_obs if next_obs is not None else obs
                agent.learn(obs, next_obs, reward, terminated, truncated, info=info)
                obs = next_obs

                episode_reward += reward
                episode_length += 1
            
            yield episode, agent.delta, episode_reward, episode_length
    
    except KeyboardInterrupt:
        pass


def main(
    # Training
    episodes: int = 10000,
    time_limit: int = 100000,
    time_limit_as_truncation: bool = False,
    # Agent
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-4,
    actor_lambda: float = 0.0,
    critic_lambda: float = 0.0,
    actor_entropy_loss_coef: float = 0.0,
    actor_hidden_layer_sizes: list[int] = [],
    critic_hidden_layer_sizes: list[int] = [],
    actor_hidden_layer_activation: nn.Module = nn.ReLU,
    critic_hidden_layer_activation: nn.Module = nn.ReLU,
    # Environment modifiers
    self_play: bool = False,
    dense_reward: bool = False,
    health: int = 3,
    use_temporal_actions: bool = False,
    observation_include_move: bool = False,
    observation_include_move_progress: bool = False,
    opponent: str = "uniform_random_play",
    # Results
    plot: bool = True,
    plot_during_training: bool = True,
    plot_save_path: str = None,
    interactive: bool = False,
    log: bool = False,
) -> tuple[list[float], list[float]]:

    if plot_during_training:
        raise ValueError("not supported until I refactor this, currently the plotter will run the training loop again")
    if plot_during_training and not plot:
        print("WARNING: specified plotting during training without plotting")
    if plot_save_path is not None and not plot:
        print("WARNING: specified a plot save path without plotting")

    torch.set_printoptions(precision=2, sci_mode=False)

    game = RPS(
        opponent=OPPONENT_POOL[opponent],
        dense_reward=dense_reward,
        health=health,
        flattened=True,
        observation_include_move=observation_include_move,
        use_temporal_actions=use_temporal_actions,
        observation_include_move_progress=observation_include_move_progress,
        observation_transformer=lambda o: o.to(torch.float32).unsqueeze(0),
        time_limit=time_limit,
        time_limit_as_truncation=time_limit_as_truncation,
    )

    agent = A2CLambdaLearner(
        discount=1.0,
        actor=ActorNetwork(
            obs_dim=game.observation_dim,
            action_dim=game.action_dim,
            hidden_layer_sizes=actor_hidden_layer_sizes,
            hidden_layer_activation=actor_hidden_layer_activation,
        ),
        critic=CriticNetwork(
            obs_dim=game.observation_dim,
            hidden_layer_sizes=critic_hidden_layer_sizes,
            hidden_layer_activation=critic_hidden_layer_activation,
        ),
        actor_lambda=actor_lambda,
        critic_lambda=critic_lambda,
        actor_entropy_loss_coef=actor_entropy_loss_coef,
        actor_optimizer=torch.optim.Adam,
        critic_optimizer=torch.optim.Adam,
        **{
            "actor_optimizer.lr": actor_lr,
            "critic_optimizer.lr": critic_lr,
        }
    )

    if self_play:
        game.set_opponent(lambda o, i: agent.sample_action(o, info=i))

    def exp_averaged_metrics(training_loop: Generator[tuple, None, None], factor: float = 0.998):
        reward_exp_avg = 0.0
        delta_exp_avg = 0.0
        episode_length_avg = 0.0
        for episode, delta, reward, episode_length in training_loop:
            reward_exp_avg = factor * reward_exp_avg + (1 - factor) * reward
            delta_exp_avg = factor * delta_exp_avg + (1 - factor) * delta
            episode_length_avg = factor * episode_length_avg + (1 - factor) * episode_length
            yield delta_exp_avg, reward_exp_avg, episode_length_avg

    training_loop = exp_averaged_metrics(train(game, agent, episodes, use_tqdm=log))
    deltas, rewards, episode_lengths = zip(*training_loop)

    if plot:
        if plot_during_training:
            animation = AnimatedPlot()
            animation.setup(training_loop)
            animation.start()
            
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
            fig: plt.Figure
            fig.set_figwidth(19)
            ax1.plot(deltas)
            ax1.set_title("Delta")
            ax1.set_xlabel("Episode")
            ax2.plot(rewards)
            ax2.set_title("Reward")
            ax2.set_xlabel("Episode")
            ax3.plot(episode_lengths)
            ax3.set_title("Episode length")
            ax3.set_xlabel("Episode")
            if plot_save_path is not None:
                plt.savefig(plot_save_path)
                plt.clf()
            else:
                plt.show()
            plt.close()
    
    else:
        for _ in training_loop:
            pass

    if log:
        print_results(game, agent)

    if interactive:
        ans = input("Rollout? ")
        while ans != "quit":
            obs, info = game.reset()
            print("Observation:", obs)
            print("Info:", info)
            print()

            terminated, truncated = False, False

            while not (terminated or truncated):
                action = agent.act(obs)
                print("Agent action:", action)
                print()

                obs, reward, terminated, truncated, info = game.step(action)
                print("Observation:", obs)
                print("Info:", info)
                print("Reward:", reward)
                print()

            print("Episode terminated\n")

            ans = input("Rollout? ")
    
    return deltas, rewards, episode_lengths


def parse_args() -> dict:
    parser = argparse.ArgumentParser("rps", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--plot", action="store_true", help="Present plots at the end of (or during) training")
    parser.add_argument("--self-play", action="store_true", help="Train the agent using self-play (naive most recent one)")
    parser.add_argument("--dense-reward", action="store_true", help="Use a dense reward scheme")
    parser.add_argument("--health", type=int, default=3, help="The health of the agents")
    parser.add_argument("--use-temporal-actions", action="store_true", help="Use temporal actions")
    parser.add_argument("--observation-include-move", action="store_true", help="Include the move information in the observation")
    parser.add_argument("--observation-include-move-progress", action="store_true", help="Include the play's move progress in the observation. Only works if using temporal actions")
    parser.add_argument("--opponent", type=str, default="uniform_random_play", choices=OPPONENT_POOL.keys(), help="The opponent to use")
    parser.add_argument("--interactive", action="store_true", help="Interactively perform rollouts with the final agent")
    parser.add_argument("--log", action="store_true", help="Print the policy and value function after training and print training progress")
    parser.add_argument("--plot-during-training", action="store_true", help="Plot the training process during training")
    parser.add_argument("--episodes", type=int, default=10000, help="The number of episodes to train for")
    parser.add_argument("--plot-save-path", type=str, default=None, help="The path to save the plot to. If None, then the plot will be shown rather than saved")

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    main(**parse_args())