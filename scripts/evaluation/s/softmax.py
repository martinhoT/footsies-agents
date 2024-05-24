from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_env_args, quick_train_args
from args import CurriculumArgs

def main(seeds: int | None = None, timesteps: int = int(1e6), processes: int = 12, y: bool = False):
    raise RuntimeError("deprecated test")

    if seeds is None:
        seeds = 6
    
    result_path = path.splitext(__file__)[0]

    runs_raw = {
        "softmax_yes": {"use_softmax": True},
        "softmax_no": {"use_softmax": False},
    }

    # Win rate against bot

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k, kwargs=v),
        env_args=quick_env_args(kwargs={"dense_reward": True}),
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
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "softmax_yes":    "Softmax",
            "softmax_no":     "No softmax",
        }
    )

    # Policy entropy

    dfs = get_data(
        data="learningpolicy_entropy",
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
        fig_path=result_path + "_ent",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Entropy",
        run_name_mapping={
            "softmax_yes":    "Softmax",
            "softmax_no":     "No softmax",
        }
    )

    # Opp model score

    dfs = get_data(
        data="learningprediction_score_of_p2s_model",
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
        fig_path=result_path + "_opp_score",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Score",
        run_name_mapping={
            "softmax_yes":    "Softmax",
            "softmax_no":     "No softmax",
        }
    )

    # Win rate against curriculum

    runs_curriculum = {k + "_curriculum": quick_train_args(
        agent_args=quick_agent_args(k + "_curriculum", kwargs=v),
        env_args=quick_env_args(
            curriculum=CurriculumArgs(
                enabled=True,
                episode_threshold=1000,
            ),
            kwargs={
                "dense_reward": True,
            }
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
        fig_path=result_path + "_curr_wr",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "softmax_yes_curriculum":    "Softmax",
            "softmax_no_curriculum":     "No softmax",
        }
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
