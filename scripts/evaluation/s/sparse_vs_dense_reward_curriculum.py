from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data, plot_add_curriculum_transitions
from scripts.evaluation.utils import quick_agent_args, quick_env_args, quick_train_args
from args import CurriculumArgs

def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 12, y: bool = False):
    result_path = path.splitext(__file__)[0]
    
    runs_raw = {
        "sparse_reward_curriculum": {"dense_reward": False},
        "dense_reward_curriculum": {"dense_reward": True},
    }

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k),
        env_args=quick_env_args(
            curriculum=CurriculumArgs(
                enabled=True,
                episode_threshold=1000,
            ),
            kwargs=v,
        ),
        timesteps=timesteps,
    ) for k, v in runs_raw.items()}

    dfs = get_data(
        data="performancewin_rate_against_current_curriculum_opponent",
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
        fig_path=None,
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "sparse_reward_curriculum":    "Sparse reward",
            "dense_reward_curriculum":     "Dense reward",
        }
    )

    # Plot vertical lines where opponent transitions occurred

    dfs_transitions = get_data(
        data="performancewin_rate_against_current_curriculum_opponent",
        runs=runs,
        seeds=seeds,
        processes=processes,
        y=y,
        data_cols=(0, 2),
    )

    if dfs_transitions is None:
        return

    plot_add_curriculum_transitions(
        dfs_transitions=dfs_transitions,
        seeds=seeds,
        fig_path=result_path + "_wr",
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)