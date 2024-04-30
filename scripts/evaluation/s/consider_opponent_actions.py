from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_env_args, quick_train_args
from args import CurriculumArgs

def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 4, y: bool = False):
    runs_raw = {
        "no_consider_opponent_actions": {"consider_opponent_at_all": False},
        "yes_consider_opponent_actions": {"consider_opponent_at_all": True},
    }
    
    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k, model="to", kwargs=v),
        env_args=quick_env_args(
            curriculum=CurriculumArgs(
                enabled=True,
                episode_threshold=1000,
            ),
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
        title="Win rate over the last 100 episodes against the in-game AI",
        fig_path=path.splitext(__file__)[0],
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "no_consider_opponent_actions":     "Consider opponent",
            "yes_consider_opponent_actions":    "Do not consider opponent",
        },
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
