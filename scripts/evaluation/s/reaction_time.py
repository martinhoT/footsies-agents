from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_train_args
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 12, y: bool = False):
    result_path = path.splitext(__file__)[0]
    
    runs_raw = {
        "reaction_correction_none":     (1,     {"use_game_model": False}),
        "reaction_correction_every_1":  (10,    {"use_game_model": True, "game_model_skippers": False, "game_model_single_skipper": 1}),
        "reaction_correction_every_3":  (4,     {"use_game_model": True, "game_model_skippers": False, "game_model_single_skipper": 3}),
        "reaction_correction_every_5":  (2.5,   {"use_game_model": True, "game_model_skippers": False, "game_model_single_skipper": 5}),
        "reaction_correction_every_15": (1,     {"use_game_model": True, "game_model_skippers": False, "game_model_single_skipper": 15}),
        "reaction_correction_skippers": (1,     {"use_game_model": True, "game_model_skippers": True, "game_model_skippers_every": 5}),
    }

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k,
            kwargs={
                "use_reaction_time": True,
                "learn": "all",
                **v,
            }
        ),
        timesteps=timesteps / timesteps_divisor,
    ) for k, (timesteps_divisor, v) in runs_raw.items()}

    # Win rate

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
        fig_path=result_path + "_wr",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "reaction_correction_none":         "No correction",
            "reaction_correction_every_1":      "1-step model",
            "reaction_correction_every_3":      "3-step model",
            "reaction_correction_every_5":      "5-step model",
            "reaction_correction_every_15":     "15-step model",
            "reaction_correction_skippers":     "Multiple models"
        }
    )

    # Reaction time

    dfs = get_data(
        data="trainingreaction_time",
        runs=runs,
        seeds=seeds,
        processes=processes,
        y=y,
    )

    if dfs is None:
        return

    plot_data(
        dfs=dfs,
        title="Reaction time against the in-game AI",
        fig_path=result_path + "_time",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Reaction time (frames)",
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
    )

    if dfs is None:
        return

    merged_df = pd.DataFrame(data={k: df["ValMean"] for k, df in dfs.items()})
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
    ax.set_title("Interaction time against the in-game AI")
    ax.set_xlabel("Model")
    ax.set_ylabel("Time (s)")
    ax.hlines(0.016, -0.5, len(runs_raw) - 0.5, colors="red", linestyles="dashed", label="Reaction hard limit")
    
    plt.xticks(rotation=30)
    
    fig = ax.get_figure()
    assert fig is not None
    fig.tight_layout()
    fig.savefig(result_path + "_act")
    fig.clf()

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
