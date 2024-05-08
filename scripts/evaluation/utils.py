import torch
from typing import Any, cast
from gymnasium import Env
from footsies_gym.envs.footsies import FootsiesEnv
from args import AgentArgs, CurriculumArgs, DIAYNArgs, EnvArgs, FootsiesSimpleActionsArgs, MainArgs, MiscArgs, SelfPlayArgs
from main import create_env


def obs_to_torch(o) -> torch.Tensor:
    return torch.from_numpy(o).float().unsqueeze(0)


def create_eval_env(
    port_start: int = 5000,
    port_stop: int | None = None,
    use_custom_opponent: bool = False,
) -> tuple[Env[torch.Tensor, int], FootsiesEnv]:
    
    ports = FootsiesEnv.find_ports(start=port_start, stop=port_stop)
    env_args = quick_env_args(kwargs=ports)

    env = create_env(env_args, use_custom_opponent=use_custom_opponent)
    footsies_env = cast(FootsiesEnv, env.unwrapped)

    return env, footsies_env


def quick_agent_args(name: str, model: str = "to", kwargs: dict[str, Any] | None = None) -> AgentArgs:
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
        "self_play": SelfPlayArgs(),
        "curriculum": CurriculumArgs(),
        
        "wrapper_time_limit": 3000,

        "footsies_wrapper_norm": True,
        "footsies_wrapper_norm_guard": True,
        "footsies_wrapper_acd": False,
        "footsies_wrapper_simple": FootsiesSimpleActionsArgs(allow_agent_special_moves=False),
        "footsies_wrapper_fs": False,
        "footsies_wrapper_adv": False,
        "footsies_wrapper_phasic": False,
        "footsies_wrapper_history": None,
    }
    root_kwargs.update(kwargs)

    inner_kwargs = {
        "game_path": "../Footsies-Gym/Build/FOOTSIES.x86_64",
        "dense_reward": False,
        "render_mode": "human",
        "sync_mode": "synced_non_blocking",
    }
    if "kwargs" in kwargs:
        inner_kwargs.update(kwargs["kwargs"])
    root_kwargs["kwargs"] = inner_kwargs

    return EnvArgs(
        name="FOOTSIES",
        torch=True,
        **root_kwargs,
    )


def quick_train_args(agent_args: AgentArgs, env_args: EnvArgs | None = None, episodes: int | None = None, timesteps: int | None = int(1e6), **kwargs) -> MainArgs:
    if env_args is None:
        env_args = quick_env_args()
    
    misc_args = MiscArgs(
        save=True,
        load=False,
        
        log_tensorboard=True,
        log_base_folder="runs",
        log_file=True,
        log_frequency=1000,
        log_test_states_number=5000,
        log_step_start=0,
        log_episode_start=0,
        log_file_level_="info",
        log_stdout_level_="info",
        
        hogwild=False,
        hogwild_cpus=None,
        hogwild_n_workers=6
    )
    
    main_kwargs = {
        "penalize_truncation": None,
        "intrinsic_reward_scheme_": None,
        "skip_freeze": True,
    }
    main_kwargs.update(kwargs)

    return MainArgs(
        misc=misc_args,
        agent=agent_args,
        env=env_args,
        episodes=episodes,
        time_steps=timesteps,
        progress_bar_kwargs={
            "desc": agent_args.name,
        },
        post_process_init=True,
        **main_kwargs,
    )

