from os import path
from scripts.evaluation.data_collectors import get_data_custom_loop, AgentCustomRun
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.custom_loop import WinRateObserver
from typing import Any, Callable
from scripts.tuning.sb3_a2c import create_model_from_parameters as create_a2c_with_parameters
from scripts.tuning.sb3_dqn import create_model_from_parameters as create_dqn_with_parameters
from scripts.tuning.sb3_ppo import create_model_from_parameters as create_ppo_with_parameters
from scripts.tuning.to import create_model_from_parameters as create_agent_with_parameters
from agents.base import FootsiesAgentBase
from stable_baselines3.common.base_class import BaseAlgorithm
from functools import partial
from gymnasium import Env


def get_best_params(study_name: str) -> dict[str, Any]:
    import optuna
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///scripts/tuning/{study_name}.db")
    return study.best_trial.params


def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 4, y: bool = False):
    result_path = path.splitext(__file__)[0]

    runs_raw: dict[str, Callable[[Env], FootsiesAgentBase | BaseAlgorithm]] = {
        "compare_ppo": partial(create_ppo_with_parameters, **get_best_params("sb3_ppo")),
        "compare_a2c": partial(create_a2c_with_parameters, **get_best_params("sb3_a2c")),
        "compare_dqn": partial(create_dqn_with_parameters, **get_best_params("sb3_dqn")),
        "compare_agent": partial(create_agent_with_parameters, **get_best_params("to")),
    }

    runs = {k: AgentCustomRun(
        agent=agent_initializer,
        opponent=None
    ) for k, agent_initializer in runs_raw.items()}

    dfs = get_data_custom_loop(
        result_path=result_path,
        runs=runs,
        observer_type=WinRateObserver,
        seeds=seeds,
        processes=processes,
        y=y,
    )

    if dfs is None:
        return
    
    plot_data(
        dfs=dfs,
        title="Win rate over the last 100 episodes against the in-game AI",
        fig_path=result_path,
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "compare_agent": "Ours",
            "compare_ppo": "PPO",
            "compare_a2c": "A2C",
            "compare_dqn": "DQN",
        }
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
