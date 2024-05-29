from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_train_args, quick_env_args
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def main(seeds: int | None = None, timesteps: int = int(1e6), processes: int = 12, y: bool = False):
    if seeds is None:
        seeds = 10

    result_path = path.splitext(__file__)[0]
    
    runs_raw = {
        "reaction_correction_every_1":  (10,    {"use_game_model": True, "game_model_skippers": False, "game_model_single_skipper": 1}),
        "reaction_correction_every_3":  (4,     {"use_game_model": True, "game_model_skippers": False, "game_model_single_skipper": 3}),
        "reaction_correction_every_5":  (2.5,   {"use_game_model": True, "game_model_skippers": False, "game_model_single_skipper": 5}),
        "reaction_correction_every_15": (1,     {"use_game_model": True, "game_model_skippers": False, "game_model_single_skipper": 15}),
        "reaction_correction_skippers": (1,     {"use_game_model": True, "game_model_skippers": True, "game_model_skippers_every": 5}),
        "reaction_correction_none":     (1,     {"use_game_model": False}),
    }

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k,
            kwargs={
                "use_reaction_time": True,
                "learn": "gm",
                "will_act_anyway": True,
                **v,
            }
        ),
        env_args=quick_env_args(kwargs={"dense_reward": True}),
        timesteps=timesteps / timesteps_divisor,
    ) for k, (timesteps_divisor, v) in runs_raw.items()}

    # Win rate

    dfs = get_data(
        data="win_rate",
        runs=runs | {
            "reaction_correction_control": quick_train_args(
                agent_args=quick_agent_args("reaction_correction_control", kwargs={"learn": "none"}),
                env_args=quick_env_args(kwargs={"dense_reward": True}),
                timesteps=timesteps
            ),
        },
        seeds=seeds,
        processes=processes,
        y=y,
        pre_trained="bot_PT",
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
            "reaction_correction_control":      "No reaction time",
            "reaction_correction_none":         "No correction",
            "reaction_correction_every_1":      "1-step model",
            "reaction_correction_every_3":      "3-step model",
            "reaction_correction_every_5":      "5-step model",
            "reaction_correction_every_15":     "15-step model",
            "reaction_correction_skippers":     "Multiple models"
        },
        ylim=(0, 1),
    )

    # Reaction time

    dfs = get_data(
        data="trainingreaction_time",
        runs=runs,
        seeds=seeds,
        processes=processes,
        y=y,
        pre_trained="bot_PT",
    )

    if dfs is None:
        return

    plot_data(
        dfs=dfs,
        title="",
        fig_path=result_path + "_time",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Average reaction time (frames)",
        run_name_mapping={
            "reaction_correction_none":         "No correction",
            "reaction_correction_every_1":      "1-step model",
            "reaction_correction_every_3":      "3-step model",
            "reaction_correction_every_5":      "5-step model",
            "reaction_correction_every_15":     "15-step model",
            "reaction_correction_skippers":     "Multiple models"
        }
    )

    # Game model loss

    runs_with_game_model = {k: v for k, v in runs.items() if k != "reaction_correction_none"}

    dfs = get_data(
        data="learninggame_model_loss",
        runs=runs_with_game_model,
        seeds=seeds,
        processes=processes,
        y=y,
        pre_trained="bot_PT",
    )

    if dfs is None:
        return

    # Since it uses 3 models, we want to see the average loss among all of them
    df = dfs["reaction_correction_skippers"]
    df["ValMean"] /= 3

    plot_data(
        dfs=dfs,
        title="",
        fig_path=result_path + "_loss_gm",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Loss",
        run_name_mapping={
            "reaction_correction_none":         "No correction",
            "reaction_correction_every_1":      "1-step model",
            "reaction_correction_every_3":      "3-step model",
            "reaction_correction_every_5":      "5-step model",
            "reaction_correction_every_15":     "15-step model",
            "reaction_correction_skippers":     "Multiple models"
        }
    )

    # Act elapsed time

    dfs = get_data(
        data="trainingact_elapsed_time_seconds",
        runs=runs,
        seeds=seeds,
        processes=processes,
        y=y,
        pre_trained="bot_PT",
    )

    if dfs is None:
        return

    merged_df = pd.DataFrame(data={k: df["ValMean"] * 1000 for k, df in dfs.items()})
    merged_df = merged_df.rename(columns={
        "reaction_correction_none":         "No correction",
        "reaction_correction_every_1":      "1-step model",
        "reaction_correction_every_3":      "3-step model",
        "reaction_correction_every_5":      "5-step model",
        "reaction_correction_every_15":     "15-step model",
        "reaction_correction_skippers":     "Multiple models"
    })

    ax = sns.boxplot(merged_df)
    ax = sns.violinplot()
    ax.set_xlabel("Model")
    ax.set_ylabel("Time (ms)")
    ax.set_ylim(bottom=0)
    # ax.hlines(16, -0.5, len(runs_raw) - 0.5, colors="red", linestyles="dashed", label="Reaction hard limit")
    
    plt.xticks(rotation=30)
    
    fig = ax.get_figure()
    assert fig is not None
    fig.tight_layout()
    fig.savefig(result_path + "_act")
    fig.clf()

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
