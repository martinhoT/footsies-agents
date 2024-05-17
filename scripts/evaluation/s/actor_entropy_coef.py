from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_add_curriculum_transitions, plot_data
from scripts.evaluation.utils import quick_agent_args, quick_env_args, quick_train_args
from args import CurriculumArgs

def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 12, y: bool = False):
    result_path = path.splitext(__file__)[0]
    
    runs_raw = {
        "actor_entropy_coef_002": {"actor_entropy_coef": 0.02},
        "actor_entropy_coef_004": {"actor_entropy_coef": 0.04},
        "actor_entropy_coef_008": {"actor_entropy_coef": 0.08},
        "actor_entropy_coef_016": {"actor_entropy_coef": 0.16},
    }
    
    # Win rate

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k, model="to", kwargs=v),
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
        fig_path=result_path + "_wr",
        exp_factor=0.9,
        xlabel="Episode",
        ylabel="Win rate",
        run_name_mapping={
            "actor_entropy_coef_002":   "$\\beta = 0.02$",
            "actor_entropy_coef_004":   "$\\beta = 0.04$",
            "actor_entropy_coef_008":   "$\\beta = 0.08$",
            "actor_entropy_coef_016":   "$\\beta = 0.16$",
        },
    )

    # Win rate against curriculum

    runs_curriculum = {k + "_curriculum": quick_train_args(
        agent_args=quick_agent_args(k + "_curriculum", model="to", kwargs=v),
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
        runs=runs_curriculum,
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
        xlabel="Episode",
        ylabel="Win rate",
        run_name_mapping={
            "actor_entropy_coef_002_curriculum":   "$\\beta = 0.02$",
            "actor_entropy_coef_004_curriculum":   "$\\beta = 0.04$",
            "actor_entropy_coef_008_curriculum":   "$\\beta = 0.08$",
            "actor_entropy_coef_016_curriculum":   "$\\beta = 0.16$",
        },
    )

    dfs_transitions = get_data(
        data="performancewin_rate_against_current_curriculum_opponent",
        runs=runs_curriculum,
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
        fig_path=result_path + "_curriculum_wr",
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
