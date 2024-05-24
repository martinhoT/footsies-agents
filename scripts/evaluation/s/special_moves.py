import torch as T
import random
from os import path
from scripts.evaluation.data_collectors import get_data, get_data_custom_loop
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_env_args, quick_train_args
from scripts.evaluation.custom_loop import WinRateObserver, AgentCustomRun, PreCustomLoop
from args import FootsiesSimpleActionsArgs, EnvArgs
from agents.the_one.agent import TheOneAgent
from footsies_gym.envs.footsies import FootsiesEnv
from gymnasium import Env
from opponents.curriculum import WhiffPunisher
from tqdm import trange
from agents.action import ActionMap
from functools import partial
from typing import cast
from models import to_
from gymnasium.spaces import Discrete


def env_with_specials(specials: bool = False) -> EnvArgs:
    return quick_env_args(
        footsies_wrapper_simple=FootsiesSimpleActionsArgs(
            enabled=True,
            allow_agent_special_moves=specials,
            assumed_agent_action_on_nonactionable="last",
            assumed_opponent_action_on_nonactionable="last"
        )
    )

def train_against_whiff_punisher(agent: TheOneAgent, env: Env, footsies_env: FootsiesEnv, initial_seed: int | None, timesteps: int = int(1e6), label: str = ""):
    T.manual_seed(initial_seed)
    random.seed(initial_seed)

    opponent = WhiffPunisher()
    footsies_env.set_opponent(opponent.act)

    seed = initial_seed
    terminated, truncated = True, True
    for _ in trange(timesteps, desc=label, unit="step", dynamic_ncols=True, colour="#FF9E64"):
        if (terminated or truncated):
            obs, info = env.reset(seed=seed)
            terminated, truncated = False, False
            seed = None

        action = agent.act(obs, info)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        reward = float(reward)

        # Skip hitstop freeze
        in_hitstop = ActionMap.is_in_hitstop_ori(next_info, True) or ActionMap.is_in_hitstop_ori(next_info, False)
        while in_hitstop and obs.isclose(next_obs).all():
            action = agent.act(obs, info)
            next_obs, r, terminated, truncated, next_info = env.step(action)
            r = float(r)
            reward += r

        agent.update(obs, next_obs, reward, terminated, truncated, info, next_info)

        obs, info = next_obs, next_info

def create_agent(env: Env) -> TheOneAgent:
    assert env.observation_space.shape
    assert isinstance(env.action_space, Discrete)

    agent, _ = to_(
        observation_space_size=env.observation_space.shape[0],
        action_space_size=int(env.action_space.n),
    )

    return agent


def main(seeds: int | None = None, timesteps: int = int(1e6), processes: int = 12, y: bool = False):
    if seeds is None:
        seeds = 3
    
    result_path = path.splitext(__file__)[0]
    
    runs_raw = {
        "no_specials": env_with_specials(False),
        "yes_specials": env_with_specials(True),
    }

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k, model="to"),
        env_args=env_args,
        timesteps=timesteps,
    ) for k, env_args in runs_raw.items()}

    # Win rate against in-game bot

    dfs = get_data(
        data="win_rate",
        runs=runs,
        seeds=seeds,
        processes=processes,
        y=y,
    )

    if dfs is None:
        return

    plot_data(
        dfs=dfs,
        title="",
        fig_path=result_path + "_wr",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "no_specials":  "Without special moves",
            "yes_specials": "With special moves",
        }
    )

    # Win rate against in-game bot after training against WhiffPunisher

    runs_custom = {k + "_pretrain": AgentCustomRun(
        agent=create_agent,
        opponent=None,
        env_args=env_args,
        pre_loop=cast(PreCustomLoop, partial(train_against_whiff_punisher, timesteps=timesteps, label=k)),
    ) for k, env_args in runs_raw.items()}

    dfs = get_data_custom_loop(
        result_path=result_path + "_pretrain",
        runs=runs_custom,
        observer_type=WinRateObserver,
        seeds=seeds,
        timesteps=timesteps,
        processes=processes,
        y=y,
    )

    if dfs is None:
        return
    
    plot_data(
        dfs=dfs,
        title="",
        fig_path=result_path + "_pretrain_wr",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "no_specials_pretrain":  "Without special moves",
            "yes_specials_pretrain": "With special moves",
        },
        attr_name="win_rate",
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
