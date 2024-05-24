from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_env_args, quick_train_args

def main(seeds: int | None = None, timesteps: int = int(2e6), processes: int = 12, y: bool = False):
    if seeds is None:
        seeds = 6
    
    runs_raw = {
        "sparse_reward": {"dense_reward": False},
        "dense_reward": {"dense_reward": True},
    }

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k),
        env_args=quick_env_args(kwargs=v),
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
        title="",
        fig_path=path.splitext(__file__)[0] + "_wr",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "sparse_reward":    "Sparse reward",
            "dense_reward":     "Dense reward",
        }
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
