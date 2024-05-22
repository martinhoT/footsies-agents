from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_env_args, quick_train_args
from args import CurriculumArgs

def main(seeds: int | None = None, timesteps: int = int(1e6), processes: int = 12, y: bool = False, do_curriculum: bool = False):
    if seeds is None:
        seeds = 6

    # Against in-game AI

    runs_raw = {
        "no_consider_opponent_actions": {"consider_opponent_at_all": False, "use_opponent_model": False, "rollback": True, "critic_opponent_update": "uniform"},
        "yes_consider_opponent_actions": {"consider_opponent_at_all": True},
    }
    
    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k, model="to", kwargs=v),
        timesteps=timesteps * 2, # we want more timesteps for when evaluating against the in-game AI to show that the same performance is still achieved, only much later
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
            "no_consider_opponent_actions":     "Do not consider opponent",
            "yes_consider_opponent_actions":    "Consider opponent",
        },
    )

    # Against curriculum

    if do_curriculum:
        runs_curriculum_raw = {
            "no_consider_opponent_actions_curriculum": {"consider_opponent_at_all": False, "use_opponent_model": False, "rollback": True, "critic_opponent_update": "uniform"},
            "yes_consider_opponent_actions_curriculum_opp": {"consider_opponent_at_all": True, "use_opponent_model": True},
            "yes_consider_opponent_actions_curriculum_perfect": {"consider_opponent_at_all": True, "use_opponent_model": False},
        }
        
        runs_curriculum = {k: quick_train_args(
            agent_args=quick_agent_args(k, model="to", kwargs=v),
            env_args=quick_env_args(
                curriculum=CurriculumArgs(
                    enabled=True,
                    episode_threshold=1000,
                ),
            ),
            timesteps=timesteps,
        ) for k, v in runs_curriculum_raw.items()}

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
            fig_path=path.splitext(__file__)[0] + "_wr_curr",
            exp_factor=0.9,
            xlabel="Episode",
            ylabel="Win rate",
            run_name_mapping={
                "no_consider_opponent_actions_curriculum":              "Do not consider opponent",
                "yes_consider_opponent_actions_curriculum_opp":         "Consider opponent (model)",
                "yes_consider_opponent_actions_curriculum_perfect":     "Consider opponent (oracle)",
            },
        )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
