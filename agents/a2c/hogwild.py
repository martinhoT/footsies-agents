import argparse
import torch
import torch.multiprocessing as mp
import gymnasium
from gymnasium.wrappers.flatten_observation import FlattenObservation
from torch import nn
from agents.a2c.a2c import A2CModule, ActorNetwork, CriticNetwork
from typing import Callable
from math import floor


EnvGenerator = Callable[[], gymnasium.Env]


def train(model: A2CModule, rank: int, env_generator: EnvGenerator, n_episodes: int, n_threads: int, base_seed: int = 1, log_interval: int = 100):
    # Set up process
    torch.manual_seed(base_seed + rank)
    # Avoid CPU oversubscription
    torch.set_num_threads(n_threads)

    # Create environment
    env = env_generator()

    # Train on environment until termination
    for episode in range(n_episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = model.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            model.update(obs, next_obs, reward, terminated)
            
            obs = next_obs
        
        if episode % log_interval == (log_interval - 1):
            print(f"Episode: {episode} | Rank: {rank}")


def hogwild(model: A2CModule, env_generator: EnvGenerator, n_episodes: int, n_processes: int, cpus_to_use: int = None, log_interval: int = 100):
    model.actor.share_memory()
    model.critic.share_memory()

    total_cpus = mp.cpu_count()
    if cpus_to_use is None:
        cpus_to_use = total_cpus
    elif cpus_to_use > total_cpus:
        raise ValueError(f"requested more CPUs than are actually available ({cpus_to_use} > {mp.cpu_count()})")

    # Set a maximum number of threads per process to avoid CPU oversubscription
    threads_per_process = floor(cpus_to_use / n_processes)

    mp.cpu_count

    processes = []
    for rank in range(n_processes):
        p = mp.Process(target=train, args=(model, rank, env_generator, n_episodes, threads_per_process, log_interval))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("hogwild", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--processes", type=int, default=4, help="number of processes on which to run Hogwild. The script will distribute threads throughout processes so as to occupy all available cores, to avoid CPU oversubscription")
    parser.add_argument("-e", "--episodes", type=int, default=2000, help="number of episodes each process will train on")
    parser.add_argument("--cpus-to-use", type=int, default=None, help="maximum number of CPUs the algorithm is allowed to use in total. If None, will use all available")
    parser.add_argument("--log-interval", type=int, default=100, help="the interval between training logs performed by each processes, in number of episodes")
    args = parser.parse_args()

    n_processes = args.processes
    n_episodes = args.episodes
    cpus_to_use = args.cpus_to_use
    log_interval = args.log_interval

    env_generator = lambda: FlattenObservation(gymnasium.make("CartPole-v1"))

    # Dummy env just to determine input and output size
    dummy_env = env_generator()
    observation_size = dummy_env.observation_space.shape[0]
    action_size = dummy_env.action_space.n

    # Hardcoded env parameters!
    model = A2CModule(
        actor=ActorNetwork(
            obs_dim=observation_size,
            action_dim=action_size,
            hidden_layer_sizes=[128],
            hidden_layer_activation=nn.ReLU,
        ),
        critic=CriticNetwork(
            obs_dim=observation_size,
            hidden_layer_sizes=[128],
            hidden_layer_activation=nn.ReLU,
        ),
        discount=0.99,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        actor_eligibility_traces_decay=0.0,
        critic_eligibility_traces_decay=0.0,
        optimizer=torch.optim.Adam,
    )

    hogwild(model, env_generator, n_episodes, n_processes, cpus_to_use, log_interval)    



