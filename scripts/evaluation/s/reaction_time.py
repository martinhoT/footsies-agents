from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_train_args

def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 4, y: bool = False):
    runs_raw = {
        "reaction_correction_none": {"use_game_model": False},
        "reaction_correction_every_1": {"use_game_model": True, "game_model_skippers": False, "game_model_single_skipper": 1},
        "reaction_correction_every_3": {"use_game_model": True, "game_model_skippers": False, "game_model_single_skipper": 3},
        "reaction_correction_every_5": {"use_game_model": True, "game_model_skippers": False, "game_model_single_skipper": 5},
        "reaction_correction_every_15": {"use_game_model": True, "game_model_skippers": False, "game_model_single_skipper": 15},
        "reaction_correction_skippers": {"use_game_model": True, "game_model_skippers": True, "game_model_skippers_every": 5},
    }

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k,
            kwargs={
                "use_reaction_time": True,
                "learn": "all",
                **v,
            }
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
            "reaction_correction_none":         "No correction",
            "reaction_correction_every_1":      "1-step model",
            "reaction_correction_every_3":      "3-step model",
            "reaction_correction_every_5":      "5-step model",
            "reaction_correction_every_15":     "15-step model",
            "reaction_correction_skippers":     "Multiple models"
        }
    )

    dfs = get_data(
        data="training/act_elapsed_time_ns",
        runs=runs,
        seeds=seeds,
        processes=processes,
        y=y,
    )

    if dfs is None:
        return

    plot_data(
        dfs=dfs,
        title="Interaction time against the in-game AI",
        fig_path=path.splitext(__file__)[0],
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Time (ns)",
        run_name_mapping={
            "reaction_correction_none":         "No correction",
            "reaction_correction_every_1":      "1-step model",
            "reaction_correction_every_3":      "3-step model",
            "reaction_correction_every_5":      "5-step model",
            "reaction_correction_every_15":     "15-step model",
            "reaction_correction_skippers":     "Multiple models"
        }
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
