from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_env_args, quick_train_args
from args import FootsiesSimpleActionsArgs

def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 12, y: bool = False):
    runs_raw = {
        "no_specials": {"remove_special_moves": True},
        "yes_specials": {"remove_special_moves": False},
    }

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k, model="to"),
        env_args=quick_env_args(
            footsies_wrapper_simple=FootsiesSimpleActionsArgs(
                enabled=True,
                allow_agent_special_moves=not v["remove_special_moves"],
                assumed_agent_action_on_nonactionable="last",
                assumed_opponent_action_on_nonactionable="last"
            )
        ),
        timesteps=timesteps,
    ) for k, v in runs_raw.items()}

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
        title="Win rate over the last 100 episodes against the in-game AI",
        fig_path=path.splitext(__file__)[0],
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "no_specials":  "Without special moves",
            "yes_specials": "With special moves",
        }
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
