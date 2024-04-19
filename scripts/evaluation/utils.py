import torch
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from functools import partial
from os import path
from collections import deque
from itertools import count, islice
from typing import Callable, Iterable, TypeVar
from agents.base import FootsiesAgentBase
from gymnasium import Env
from gymnasium.wrappers.transform_observation import TransformObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation
from agents.wrappers import FootsiesSimpleActions
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers import FootsiesNormalized
from agents.the_one.agent import TheOneAgent
from agents.action import ActionMap
from tqdm import trange
from args import AgentArgs, DIAYNArgs, EnvArgs, MainArgs, MiscArgs, SelfPlayArgs
from main import main


def batched(iterable: Iterable, n: int):
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def dummy_opponent(o: dict, i: dict) -> tuple[bool, bool, bool]:
    return (False, False, False)

def obs_to_torch(o) -> torch.Tensor:
    return torch.from_numpy(o).float().unsqueeze(0)

def create_env(
    port_start: int = 5000,
    port_stop: int | None = None,
) -> tuple[Env, FootsiesEnv]:
    
    footsies_env = FootsiesEnv(
        game_path="../Footsies-Gym/Build/FOOTSIES.x86_64",
        render_mode=None,
        sync_mode="synced_non_blocking",
        fast_forward=True,
        dense_reward=False,
        opponent=dummy_opponent,
        **FootsiesEnv.find_ports(start=port_start, stop=port_stop),
    )

    env = TransformObservation(
        FootsiesSimpleActions(
            FlattenObservation(
                FootsiesNormalized(
                    footsies_env,
                )
            )
        ),
        obs_to_torch,
    )

    return env, footsies_env


class Observer:
    def update(self, step: int, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: FootsiesAgentBase):
        pass

    @property
    def data(self) -> tuple[list[int], list[float]]:
        raise NotImplementedError("this observer didn't implement the 'data' property")


class WinRateObserver(Observer):
    def __init__(self, last: int = 100):
        self._wins = deque([], maxlen=last)
        self._win_rates = []

    def update(self, step: int, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: FootsiesAgentBase):
        if terminated or truncated:
            won = (reward > 0) if terminated else (next_info["guard"][0] > next_info["guard"][1])
            self._wins.append(won)
            self._win_rates.append(sum(self._wins) / len(self._wins))

    @property
    def data(self) -> tuple[list[int], list[float]]:
        return list(range(len(self._win_rates))), self._win_rates


T = TypeVar("T", bound=Observer)

def test(agent: TheOneAgent, label: str, id_: int, observer_type: type[T], opponent: Callable[[dict, dict], tuple[bool, bool, bool]] | None = None, initial_seed: int | None = 0, timesteps: int = 1000000) -> list[T]:
    port_start = 11000 + 1000 * id_
    port_stop = 11000 + (1000) * (id_ + 1)
    env, footsies_env = create_env(port_start=port_start, port_stop=port_stop)

    footsies_env.set_opponent(opponent)

    observer = observer_type()

    seed = initial_seed
    terminated, truncated = True, True
    for step in trange(timesteps, desc=label, unit="step", position=id_, dynamic_ncols=True, colour="#80e4ed"):
        if (terminated or truncated):
            obs, info = env.reset(seed=seed)
            terminated, truncated = False, False
            seed = None

        action = agent.act(obs, info)
        next_obs, reward, terminated, truncated, next_info = env.step(action)

        # Skip hitstop freeze
        in_hitstop = ActionMap.is_in_hitstop_ori(next_info, True) or ActionMap.is_in_hitstop_ori(next_info, False)
        while in_hitstop and obs.isclose(next_obs).all():
            action = agent.act(obs, info)
            next_obs, r, terminated, truncated, next_info = env.step(action)
            reward += r

        agent.update(obs, next_obs, reward, terminated, truncated, info, next_info)
        observer.update(step, obs, next_obs, reward, terminated, truncated, info, next_info, agent)

        obs, info = next_obs, next_info
    
    return observer


def get_data_custom_loop(result_path: str, agents: list[tuple[str, FootsiesAgentBase]], observer_type: type[T], opponent: Callable[[dict, dict], tuple[bool, bool, bool]] | None = None, seeds: int = 10, timesteps: int = 1000000) -> dict[str, pd.DataFrame] | None:
    dfs = {}
    for name, _ in agents:
        df_path = f"{result_path}_{name}"
        if path.exists(df_path):
            dfs[name] = pd.read_csv(df_path)
        else:
            break

    require_collection = len(dfs) != len(agents)

    if require_collection:
        ans = input("Will require collection of data, proceed? [y/N] ")
        if ans.upper() != "Y":
            return None

        # Collect the data

        seeds = 10
        with mp.Pool(processes=4) as pool:
            test_partial = partial(test, timesteps=timesteps)
            runs = [
                (agent, name, i, observer_type, opponent.act, seed)
                for i, (name, agent) in enumerate(agents)
                for seed in range(seeds)
            ]
            observers: list[Observer] = pool.starmap(test_partial, runs)
        
        observers: list[list[Observer]] = list(batched(observers, seeds))

        # Create dataframes with the data

        seed_columns = [f"Val{seed}" for seed in range(seeds)]
        dfs: dict[str, pd.DataFrame] = {
            name: pd.DataFrame(
                data=[agent_observers[0].data[0]] + [o.data[1] for o in agent_observers],
                columns=["Idx"] + seed_columns
            ) for (name, _), agent_observers in zip(agents, observers)
        }

        for _, df in dfs.items():
            df["ValMean"] = df[seed_columns].mean(axis=1)
            df["ValStd"] = df[seed_columns].std(axis=1)
        
        # Save the data for posterity

        for name, df in dfs.items():
            df.to_csv(f"{result_path}_{name}")


def quick_agent_args(name: str, **kwargs) -> AgentArgs:
    return AgentArgs(
        model="to",
        kwargs=kwargs,
        name=name,
        is_sb3=False,
        post_process_init=True,
    )


def quick_env_args(**kwargs) -> EnvArgs:
    base_kwargs = {
        "game_path": "../Footsies-Gym/Build/FOOTSIES.x86_64",
        "dense_reward": False,
    }
    base_kwargs.update(kwargs)
    kwargs = base_kwargs

    return EnvArgs(
        diayn=DIAYNArgs(),
        
        name="FOOTSIES",
        kwargs=kwargs,
        wrapper_time_limit=3000,
        
        footsies_wrapper_norm=True,
        footsies_wrapper_norm_guard=True,
        footsies_wrapper_acd=False,
        footsies_wrapper_simple=(True, True, "last", "last"),
        footsies_wrapper_fs=False,
        footsies_wrapper_adv=False,
        footsies_wrapper_phasic=False,
        footsies_wrapper_history=None,
        
        torch=True,
        log_dir="runs",
    )


def quick_train_args(seed: int, agent_args: AgentArgs, env_args: EnvArgs, episodes: int | None = None, timesteps: int | None = 2500000, self_play_args: SelfPlayArgs | None = None, **kwargs) -> MainArgs:
    misc_args = MiscArgs(
        save=True,
        load=False,
        
        log_tensorboard=True,
        log_file=False,
        log_frequency=10000,
        log_test_states_number=5000,
        log_step_start=0,
        log_episode_start=0,
        log_file_level_="info",
        log_stdout_level_="info",
        
        hogwild=False,
        hogwild_cpus=None,
        hogwild_n_workers=6
    )
    
    self_play_args = self_play_args if self_play_args is not None else SelfPlayArgs()

    main_kwargs = {
        "penalize_truncation": None,
        "curriculum": False,
        "intrinsic_reward_scheme_": None,
        "skip_freeze": True,
    }
    main_kwargs.update(kwargs)

    return MainArgs(
        misc=misc_args,
        agent=agent_args,
        env=env_args,
        self_play=self_play_args,
        episodes=episodes,
        time_steps=timesteps,
        seed=seed,
        progress_bar_kwargs={
            "desc": agent_args.name,
        },
        post_process_init=True,
        **main_kwargs,
    )


def get_data(data: str, agents: list[tuple[str, dict, dict, dict]], seeds: int = 10, timesteps: int | None = 2500000) -> dict[str, pd.DataFrame] | None:
    missing = []
    for agent in agents:
        name, agent_kwargs, env_kwargs, main_kwargs = agent
        for seed in range(seeds):
            full_name = f"eval_{name}_S{seed}"
            data_path = path.join("runs", full_name)
            if not path.exists(data_path):
                missing.append((full_name, agent_kwargs, env_kwargs, main_kwargs))

    if missing:
        names_list = [f"- {full_name}" for full_name, _, _, _ in missing]
        print("The following runs are missing:", *names_list, sep="\n")
        ans = input("Do you want to run them now? [y/N] ")
        if ans.upper() != "Y":
            return None
        
        # Try to avoid oversubscription, since each agent will have at least 2 processes running: itself, and the game
        with mp.Pool(processes=4) as pool:
            args = [
                quick_train_args(
                    seed=seed,
                    agent_args=quick_agent_args(
                        name=full_name, 
                        **agent_kwargs,
                    ),
                    env_args=quick_env_args(
                        **env_kwargs,
                        **FootsiesEnv.find_ports(11000 + i*25, stop=11000 + i*1000 + (i+1)*25)
                    ),
                    episodes=None,
                    timesteps=timesteps,
                    self_play_args=None,
                    **main_kwargs,
                )
                for i, (full_name, agent_kwargs, env_kwargs, main_kwargs) in enumerate(missing)
            ]
            
            pool.map(main, args)
    
    dfs: dict[str, pd.DataFrame] = {}
    for name, _, _, _ in agents:
        df = pd.DataFrame([], columns=["Idx", "ValMean", "ValStd"])
        seed_columns = [f"Val{seed}" for seed in range(seeds)]

        for seed in range(seeds):
            data_path = path.join("runs", f"eval_{name}_S{seed}", f"{data}.csv")
            d = pd.read_csv(data_path, names=["Idx", "Val"])
            df["Idx"] = d["Idx"]
            df[f"Val{seed}"] = d["Val"]
        
        df["ValMean"] = df[seed_columns].mean(axis=1)
        df["ValStd"] = df[seed_columns].std(axis=1)

        dfs[name] = df
    
    return dfs


def plot_data(dfs: dict[str, pd.DataFrame], title: str, fig_path: str, exp_factor: float = 0.9, xlabel: str | None = None, ylabel: str | None = None, run_name_mapping: dict[str, str] | None = None):
    # Smooth the values (make exponential moving average) and plot them
    alpha = 1 - exp_factor
    for df in dfs.values():
        df["ValMeanExp"] = df["ValMean"].ewm(alpha=alpha).mean()
        df["ValStdExp"] = df["ValStd"].ewm(alpha=alpha).mean()
        plt.plot(df.Idx, df.ValMeanExp)

    for df in dfs.values():
        plt.fill_between(df.Idx, df.ValMeanExp - df.ValStdExp, df.ValMeanExp + df.ValStdExp, alpha=0.2)

    if run_name_mapping is not None:
        plt.legend([run_name_mapping[name] for name in dfs.keys()])
    plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.savefig(fig_path)


def get_and_plot_data(
    data: str,
    agents: tuple[str, dict, dict, dict],
    title: str,
    fig_path: str,
    seeds: int = 10,
    timesteps: int | None = 2500000,
    exp_factor: float = 0.9,
    xlabel: str | None = None,
    ylabel: str | None = None,
    run_name_mapping: dict[str, str] | None = None
):
    dfs = get_data(
        data=data,
        agents=agents,
        seeds=seeds,
        timesteps=timesteps
    )

    if dfs is None:
        print("Did not get the data, quitting")
        exit(0)

    plot_data(
        dfs=dfs,
        title=title,
        fig_path=fig_path,
        exp_factor=exp_factor,
        xlabel=xlabel,
        ylabel=ylabel,
        run_name_mapping=run_name_mapping
    )
