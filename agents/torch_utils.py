import torch
import torch.multiprocessing as mp
from typing import Callable
from math import floor
from torch import nn
from itertools import pairwise
from gymnasium import Env

from agents.base import FootsiesAgentTorch
from agents.logger import TrainingLoggerWrapper


def create_layered_network(
    input_dim: int,
    output_dim: int,
    hidden_layer_sizes: list[int],
    hidden_layer_activation: nn.Module,
):
    """Create a multi-layered network. Each integer in `hidden_layer_sizes` represents a layer of the given size. `hidden_layer_activation` is the activation function applied at every hidden layer"""
    if hidden_layer_sizes is None:
        hidden_layer_sizes = []

    if len(hidden_layer_sizes) == 0:
        layers = nn.Sequential(nn.Linear(input_dim, output_dim))
    else:
        layers = [
            nn.Linear(input_dim, hidden_layer_sizes[0]),
            hidden_layer_activation(),
        ]

        for hidden_layer_size_in, hidden_layer_size_out in pairwise(hidden_layer_sizes):
            layers.append(
                nn.Linear(hidden_layer_size_in, hidden_layer_size_out)
            )
            layers.append(
                hidden_layer_activation(),
            )
        
        layers.append(
            nn.Linear(hidden_layer_sizes[-1], output_dim)
        )

        layers = nn.Sequential(*layers)

    return layers


def hogwild(
    agent: FootsiesAgentTorch,
    env_generator: Callable[..., Env],
    training_method: Callable,
    n_workers: int,
    cpus_to_use: int = None,
    is_footsies: bool = False,
    logging_wrapper: Callable[[FootsiesAgentTorch], TrainingLoggerWrapper] = None,
    **training_method_kwargs
):
    """
    Wrapper on a training method, which will train a PyTorch-based agent using the Hogwild algorithm.
    Each worker is assigned a unique rank.
    
    Parameters
    ----------
    - `agent`: implementation of FootsiesAgentTorch, representing the agent
    - `env_generator`: callable that will generate a Gymnasium environment to train on. Accepts keyword arguments
    - `training_method`: the method to use for training the agent
    - `n_workers`: the number of parallel workers to use for training
    - `cpus_to_use`: the maximum number of CPUs to use for training
    - `is_footsies`: whether the environment generator generates a FOOTSIES environment. This is required to prepare FOOTSIES-specific parameters
    - `logging_wrapper`: the logging wrapper to apply to the worker of rank 0. If None, no logging will be performed
    - `training_method_kwargs`: additional keyword arguments to pass to the training method
    """
    agent.model.share_memory()

    total_cpus = mp.cpu_count()
    if cpus_to_use is None:
        cpus_to_use = total_cpus
    elif cpus_to_use > total_cpus:
        raise ValueError(f"requested more CPUs than are actually available ({cpus_to_use} > {mp.cpu_count()})")

    # Set a maximum number of threads per process to avoid CPU oversubscription.
    # Leave 1 for the FOOTSIES game itself, if we are training on FOOTSIES.
    if is_footsies:
        threads_per_process = floor(cpus_to_use / n_workers) - 1
    else:
        threads_per_process = floor(cpus_to_use / n_workers)

    if threads_per_process <= 0:
        raise ValueError(f"the number of processes is too large, having {floor(cpus_to_use / n_workers)} threads per worker is not enough")

    def worker_train(
        agent: FootsiesAgentTorch,
        env_generator: Callable[..., Env],
        rank: int,
        n_threads: int = 1,
        base_seed: int = 1,
        is_footsies: bool = False,
        logging_wrapper: Callable[[FootsiesAgentTorch], TrainingLoggerWrapper] = None,
        **train_kwargs,
    ):
        # Set up process
        torch.manual_seed(base_seed + rank)
        # Avoid CPU oversubscription
        torch.set_num_threads(n_threads)

        if logging_wrapper is not None and rank == 0:
            agent = logging_wrapper(agent)

        # We need to set different FOOTSIES instances with different ports for each worker
        if is_footsies:
            from psutil import net_connections
            closed_ports = {p.laddr.port for p in net_connections(kind="tcp4")}

            base_port = 10000 + rank * 1000

            # We need a triplet of ports
            ports = []
            for port in range(base_port, base_port + 1000):
                if port not in closed_ports:
                    ports.append(port)

                if len(ports) >= 3:
                    break
            
            if len(ports) < 3:
                raise RuntimeError(f"could not find 3 free ports for worker {rank}'s FOOTSIES instance")

            print(f"[{rank}] I was assigned the ports: {ports}")
            
            env = env_generator(
                game_port=ports[0],
                opponent_port=ports[1],
                remote_control_port=ports[2],
                log_file="out.log" if rank == 0 else None,
                log_file_overwrite=True,
            )
        
        else:
            env = env_generator()

        def log_episode(episode):
            if episode > 0 and episode % 100 == 0:
                print(f"[{rank}] Reached episode {episode}")

        # Overwrite keyword arguments with values that only make sense for asynchronous training
        kwargs_that_shouldnt_be_set = {"episode_finished_callback", "progress_bar"}
        kwargs_that_were_set_but_shouldnt_be = kwargs_that_shouldnt_be_set & train_kwargs.keys()
        if kwargs_that_were_set_but_shouldnt_be:
            print(f"[{rank}] WARNING: I was supplied with training kwargs that shouldn't have been set ({kwargs_that_were_set_but_shouldnt_be}), will overwrite them")
        train_kwargs["episode_finished_callback"] = log_episode
        train_kwargs["progress_bar"] = False

        print(f"[{rank}] Started training")
        training_method(agent, env, **train_kwargs)

    processes = []
    for rank in range(n_workers):
        p = mp.Process(target=worker_train, args=(agent, env_generator), kwargs={
            "rank": rank,
            "n_threads": threads_per_process,
            "base_seed": 1,
            "is_footsies": is_footsies,
            "logging_wrapper": logging_wrapper,
            **training_method_kwargs,
        })
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


class InputClip(nn.Module):
    # Range of [5, 5] allows representing the sigmoid values of 0.01 and 0.99
    # The idea is that the network should not be too sure or unsure of the outcomes, and allow better adaptability by avoiding very small gradients at the sigmoid's tails
    # NOTE: watch out, gradients can become 0 without leaking, a leaky version has better adaptability (as in the Loss of Plasticity in Deep Continual Learning paper)
    def __init__(self, minimum: float = -5, maximum: float = 5, leaky_coef: float = 0):
        """Clip input into range"""
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum
        self.leaky_coef = leaky_coef

    def forward(self, x: torch.Tensor):
        return torch.clip(
            x,
            min=self.minimum + self.leaky_coef * (x - 5),
            max=self.maximum + self.leaky_coef * (x + 5),
        )


class ProbabilityDistribution(nn.Module):
    def __init__(self):
        """Makes input sum to 1"""
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x / torch.sum(x)


class DebugStoreRecent(nn.Module):
    def __init__(self):
        """Store the most recent input"""
        super().__init__()
        self.stored = None

    def forward(self, x: torch.Tensor):
        self.stored = x
        return x


class AggregateModule(nn.Module):
    def __init__(self, modules: dict[str, nn.Module]):
        """
        Aggregate of multiple modules.
        Useful for reporting a model for implementations of `FootsiesAgentTorch`, when multiple modules are used.
        Only works when all modules accept a single argument in `forward()`.
        """
        super().__init__()
        for name, module in modules.items():
            self.add_module(name, module)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return tuple(module(x) for module in self.children())


def observation_invert_perspective_flattened(obs: torch.Tensor) -> torch.Tensor:
    inverted = obs.clone().detach()
    
    #  guard
    inverted[:, [0, 1]] = obs[:, [1, 0]]
    #  move
    inverted[:, 2:17] = obs[:, 17:32]
    inverted[:, 17:32] = obs[:, 2:17]
    #  move progress
    inverted[:, [32, 33]] = obs[:, [33, 32]]
    #  position
    inverted[:, [34, 35]] = obs[:, [35, 34]]

    return inverted


def hidden_layer_parameters_from_specifications(
    hidden_layer_sizes_specification: str,
    hidden_layer_activation_specification: str,
) -> tuple[list[int], type[nn.Module]]:
    
    hidden_layer_sizes = [int(n) for n in hidden_layer_sizes_specification.split(",")] if hidden_layer_sizes_specification else []
    hidden_layer_activation = getattr(nn, hidden_layer_activation_specification)

    return hidden_layer_sizes, hidden_layer_activation
