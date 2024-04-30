from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_env_args, quick_train_args
from args import FootsiesSimpleActionsArgs
from typing import Literal

def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 4, y: bool = False):
    runs_raw: dict[str, Literal["last", "none", "stand"]] = {
        "assumed_opponent_action_on_frameskip_last": "last",
        "assumed_opponent_action_on_frameskip_none": "none",
        "assumed_opponent_action_on_frameskip_stand": "stand",
    }
    
    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k),
        env_args=quick_env_args(
            footsies_wrapper_simple=FootsiesSimpleActionsArgs(
                enabled=True,
                allow_agent_special_moves=True,
                assumed_agent_action_on_nonactionable="last",
                assumed_opponent_action_on_nonactionable=assumed_action,
            )
        ),
        timesteps=timesteps,
    ) for k, assumed_action in runs_raw.items()}

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
        title="Win rate over the last 100 episodes against the in-game bot",
        fig_path=path.splitext(__file__)[0],
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "assumed_opponent_action_on_frameskip_last":    "Last",
            "assumed_opponent_action_on_frameskip_none":    "None",
            "assumed_opponent_action_on_frameskip_stand":   "No-op",
        },
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
