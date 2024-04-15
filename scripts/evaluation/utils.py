import torch
import pandas as pd
import multiprocessing as mp
from os import path
from typing import Callable, TypeVar
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
    def update(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: FootsiesAgentBase):
        pass


T = TypeVar("T", bound=Observer)

def test(agent: TheOneAgent, label: str, id_: int, observer_type: type[T], opponent: Callable[[dict, dict], tuple[bool, bool, bool]] | None = None, episodes: int = 1000) -> T:
    port_start = 11000 + 100 * id_
    port_stop = 11000 + (100) * (id_ + 1)
    env, footsies_env = create_env(port_start=port_start, port_stop=port_stop)

    footsies_env.set_opponent(opponent)

    observer = observer_type()

    for _ in trange(episodes, desc=label, unit="ep", position=id_, dynamic_ncols=True, colour="#80e4ed"):
        obs, info = env.reset()
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = agent.act(obs, info)
            next_obs, reward, terminated, truncated, next_info = env.step(action)

            # Skip hitstop freeze
            in_hitstop = ActionMap.is_in_hitstop_ori(next_info, True) or ActionMap.is_in_hitstop_ori(next_info, False)
            while in_hitstop and obs.isclose(next_obs).all():
                action = agent.act(obs, info)
                next_obs, r, terminated, truncated, next_info = env.step(action)
                reward += r

            agent.update(obs, next_obs, reward, terminated, truncated, info, next_info)
            observer.update(obs, next_obs, reward, terminated, truncated, info, next_info, agent)

            obs, info = next_obs, next_info
    
    return observer


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


def quick_train_args(seed: int, agent_args: AgentArgs, env_args: EnvArgs, episodes: int | None = None, timesteps: int | None = 2500000, self_play_args: SelfPlayArgs | None = None) -> MainArgs:
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

    return MainArgs(
        misc=misc_args,
        agent=agent_args,
        env=env_args,
        self_play=self_play_args,
        episodes=episodes,
        time_steps=timesteps,
        penalize_truncation=None,
        curriculum=False,
        intrinsic_reward_scheme_=None,
        skip_freeze=True,
        seed=seed,
        progress_bar_kwargs={
            "desc": agent_args.name,
        },
        post_process_init=True,
    )


def get_data(data: str, agents: tuple[str, dict, dict], seeds: int = 10, timesteps: int = 2500000) -> dict[str, pd.DataFrame]:
    missing = []
    for agent in agents:
        name, agent_kwargs, env_kwargs = agent
        for seed in range(seeds):
            full_name = f"eval_{name}_S{seed}"
            data_path = path.join("runs", full_name)
            if not path.exists(data_path):
                missing.append((full_name, agent_kwargs, env_kwargs))

    if missing:
        names_list = [f"- {full_name}" for full_name, _, _ in missing]
        print("The following runs are missing:", *names_list, sep="\n")
        ans = input("Do you want to run them now? [y/N] ")
        if ans.upper() != "Y":
            print("Did not accept to get missing data, quitting")
            return None
        
        # Try to avoid oversubscription, since each agent will have at least 2 processes running: itself, and the game
        with mp.Pool(processes=6) as pool:
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
                )
                for i, (full_name, agent_kwargs, env_kwargs) in enumerate(missing)
            ]
            
            pool.map(main, args)
    
    dfs: dict[str, pd.DataFrame] = {}
    for name, _, _ in agents:
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