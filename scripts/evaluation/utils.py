import torch
from itertools import islice
from typing import Iterable, Any
from gymnasium import Env
from gymnasium.wrappers.transform_observation import TransformObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation
from agents.wrappers import FootsiesSimpleActions
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers import FootsiesNormalized
from args import AgentArgs, DIAYNArgs, EnvArgs, MainArgs, MiscArgs, SelfPlayArgs


def dummy_opponent(o: dict, i: dict) -> tuple[bool, bool, bool]:
    return (False, False, False)


def obs_to_torch(o) -> torch.Tensor:
    return torch.from_numpy(o).float().unsqueeze(0)


def create_env(
    port_start: int = 5000,
    port_stop: int | None = None,
) -> tuple[Env[torch.Tensor, int], FootsiesEnv]:
    
    footsies_env = FootsiesEnv(
        game_path="../Footsies-Gym/Build/FOOTSIES.x86_64",
        render_mode=None,
        sync_mode="synced_non_blocking",
        fast_forward=True,
        dense_reward=False,
        opponent=dummy_opponent,
        **FootsiesEnv.find_ports(start=port_start, stop=port_stop), # type: ignore
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


def quick_agent_args(name: str, model: str = "to", kwargs: dict | None = None) -> AgentArgs:
    if kwargs is None:
        kwargs = {}
    
    return AgentArgs(
        model=model,
        kwargs=kwargs,
        name_=name,
        is_sb3=False,
        post_process_init=True,
    )


def quick_env_args(**kwargs) -> EnvArgs:
    root_kwargs: dict[str, Any] = {
        "diayn": DIAYNArgs(),
        
        "wrapper_time_limit": 3000,

        "footsies_wrapper_norm": True,
        "footsies_wrapper_norm_guard": True,
        "footsies_wrapper_acd": False,
        "footsies_wrapper_simple": (True, True, "last", "last"),
        "footsies_wrapper_fs": False,
        "footsies_wrapper_adv": False,
        "footsies_wrapper_phasic": False,
        "footsies_wrapper_history": None,
    }
    root_kwargs.update(kwargs)

    inner_kwargs = {
        "game_path": "../Footsies-Gym/Build/FOOTSIES.x86_64",
        "dense_reward": False,
    }
    inner_kwargs.update(kwargs["kwargs"])
    root_kwargs["kwargs"] = inner_kwargs

    return EnvArgs(
        name="FOOTSIES",
        torch=True,
        log_dir="runs",
        **root_kwargs,
    )


def quick_train_args(agent_args: AgentArgs, env_args: EnvArgs | None = None, episodes: int | None = None, timesteps: int | None = int(1e6), self_play_args: SelfPlayArgs | None = None, **kwargs) -> MainArgs:
    if env_args is None:
        env_args = quick_env_args()
    
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
        progress_bar_kwargs={
            "desc": agent_args.name,
        },
        post_process_init=True,
        **main_kwargs,
    )

