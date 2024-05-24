from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_curriculum, plot_data, plot_add_curriculum_transitions
from scripts.evaluation.utils import quick_agent_args, quick_env_args, quick_train_args
from args import CurriculumArgs

def main(seeds: int | None = None, timesteps: int | None = None, processes: int = 12, y: bool = False):
    if seeds is None:
        seeds = 3
    
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
                episode_threshold=10000,
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

    plot_curriculum(
        dfs=dfs,
        dfs_transitions=dfs_transitions,
        seeds=seeds,
        title="",
        fig_path=result_path,
        exp_factor=0.9,
        ylabel="Win rate",
        run_name_mapping={
            "sparse_reward_curriculum":    "Sparse reward",
            "dense_reward_curriculum":     "Dense reward",
        }
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)