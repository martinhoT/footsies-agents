from os import path
from scripts.evaluation.data_collectors import get_data_custom_loop
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.custom_loop import WinRateObserver, AgentCustomRun
from scripts.evaluation.utils import quick_env_args
from typing import Any, Callable
from scripts.tuning.sb3_a2c import create_model_from_parameters as create_a2c_with_parameters
from scripts.tuning.sb3_dqn import create_model_from_parameters as create_dqn_with_parameters
from scripts.tuning.sb3_ppo import create_model_from_parameters as create_ppo_with_parameters
from scripts.tuning.to import create_model_from_parameters as create_agent_with_parameters
from agents.base import FootsiesAgentBase
from stable_baselines3.common.base_class import BaseAlgorithm
from functools import partial
from gymnasium import Env
from models import to_
from gymnasium.spaces import Discrete
from agents.the_one.agent import TheOneAgent


def get_best_params(study_name: str) -> dict[str, Any]:
    import optuna
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///scripts/tuning/{study_name}.db")
    return study.best_trial.params


def create_the_one_default(env: Env) -> TheOneAgent:
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

    runs_raw: dict[str, Callable[[Env], FootsiesAgentBase | BaseAlgorithm]] = {
        "agent": partial(create_agent_with_parameters, **get_best_params("to")),
        # "agent": create_the_one_default,
        "ppo": partial(create_ppo_with_parameters, **get_best_params("sb3_ppo")),
        "a2c": partial(create_a2c_with_parameters, **get_best_params("sb3_a2c")),
        "dqn": partial(create_dqn_with_parameters, **get_best_params("sb3_dqn")),
    }

    env_args = quick_env_args(kwargs={"dense_reward": True})
    runs = {k: AgentCustomRun(
        agent=agent_initializer,
        opponent=None,
        env_args=env_args,
    ) for k, agent_initializer in runs_raw.items()}

    dfs = get_data_custom_loop(
        result_path=result_path,
        runs=runs,
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
        fig_path=result_path,
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "agent": "Ours",
            "ppo": "PPO",
            "a2c": "A2C",
            "dqn": "DQN",
        },
        attr_name="win_rate"
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
