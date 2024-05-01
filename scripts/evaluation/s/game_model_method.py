from os import path
from models import game_model_
from scripts.evaluation.data_collectors import get_data, get_data_dataset
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_train_args, create_eval_env
from scripts.evaluation.custom_loop import GameModelObserver
from gymnasium.spaces import Discrete

def main(seeds: int = 10, timesteps: int = int(1e6), epochs: int = 10, processes: int = 4, shuffle: bool = True, name_suffix: str = "", y: bool = False):
    runs_raw = {
        "gm_residual": {"learn": "all", "game_model_method": "residual", "game_model_skippers": False},
        "gm_normal": {"learn": "all", "game_model_method": "normal", "game_model_skippers": False},
        "gm_differences": {"learn": "all", "game_model_method": "differences", "game_model_skippers": False},
    }

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k, model="to", kwargs=v),
        timesteps=timesteps,
    ) for k, v in runs_raw.items()}

    result_basename = path.splitext(__file__)[0] + name_suffix

    # Win rate on normal agent
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
        title="Win rate over the last 100 episodes against the in-game bot",
        fig_path=result_basename + "_wr",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "gm_residual":     "Residual",
            "gm_normal":       "Normal",
            "gm_differences":  "Differences",
        }
    )

    # Losses on normal agent
    for label in ["", "guard", "move", "move_progress", "position"]:
        data_label = ("_" + label) if label else label
        title_label = f" ({label})" if label else label

        dfs = get_data(
            data=f"learninggame_model_loss{data_label}",
            runs=runs,
            seeds=seeds,
            processes=processes,
        )

        if dfs is None:
            return

        plot_data(
            dfs=dfs,
            title=f"Game model loss against the in-game bot{title_label}",
            fig_path=f"{result_basename}_gm{data_label}",
            exp_factor=0.9,
            xlabel="Time step",
            ylabel="Loss",
            run_name_mapping={
                "gm_residual":     "Residual",
                "gm_normal":       "Normal",
                "gm_differences":  "Differences",
            }
        )

    # Losses on dataset
    runs_dataset_raw = {
        "gm_residual_gm": {"residual": True, "by_differences": False},
        "gm_normal_gm": {"residual": False, "by_differences": False},
        "gm_differences_gm": {"residual": False, "by_differences": True},
    }
    
    dummy_env, _ = create_eval_env()
    assert dummy_env.observation_space.shape
    assert isinstance(dummy_env.action_space, Discrete)

    for label in ["", "guard", "move", "move_progress", "position"]:
        data_label = ("_" + label) if label else label
        title_label = f" ({label})" if label else label

        runs_dataset = {
            k: game_model_(**v)[0]
            for k, v in runs_dataset_raw.items()
        }

        dfs = get_data_dataset(
            result_path=result_basename,
            runs=runs_dataset,
            observer_type=GameModelObserver,
            seeds=seeds,
            processes=processes,
            epochs=epochs,
            shuffle=shuffle,
        )

        if dfs is None:
            return
        
        plot_data(
            dfs=dfs,
            title=f"Game model loss on the dataset{title_label}",
            fig_path=f"{result_basename}_gm_dataset{data_label}",
            exp_factor=0.9,
            xlabel="Time step",
            ylabel="Loss",
            run_name_mapping={
                "gm_residual_gm":     "Residual",
                "gm_normal_gm":       "Normal",
                "gm_differences_gm":  "Differences",
            },
            attr_name="loss" + data_label,
        )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)