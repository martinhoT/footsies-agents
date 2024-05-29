from os import path
from models import game_model_
from scripts.evaluation.data_collectors import get_data, get_data_dataset
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_train_args, create_eval_env
from scripts.evaluation.custom_loop import GameModelObserver
from gymnasium.spaces import Discrete

def main(seeds: int | None = None, timesteps: int = int(1e6), epochs: int = 10, processes: int = 12, shuffle: bool = True, name_suffix: str = "", y: bool = False):
    if seeds is None:
        seeds = 10
    
    runs_raw = {
        "gm_residual": {"learn": "gm", "game_model_method": "residual", "game_model_skippers": True, "use_reaction_time": True, "will_act_anyway": True},
        "gm_normal": {"learn": "gm", "game_model_method": "normal", "game_model_skippers": True, "use_reaction_time": True, "will_act_anyway": True},
        "gm_differences": {"learn": "gm", "game_model_method": "differences", "game_model_skippers": True, "use_reaction_time": True, "will_act_anyway": True},
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
        pre_trained="bot_PT",
    )

    if dfs is None:
        return

    plot_data(
        dfs=dfs,
        title="",
        fig_path=result_basename + "_wr",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "gm_residual":     "Residual",
            "gm_normal":       "Normal",
            "gm_differences":  "Differences",
        },
        ylim=(0, 1),
    )

    # Losses on normal agent
    for label in ["", "guard", "move", "move_progress", "position"]:
        data_label = ("_" + label) if label else label

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
            title="",
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
        "dataset_gm_residual": {"residual": True, "by_differences": False},
        "dataset_gm_normal": {"residual": False, "by_differences": False},
        "dataset_gm_differences": {"residual": False, "by_differences": True},
    }
    
    dummy_env, _ = create_eval_env()
    assert dummy_env.observation_space.shape
    assert isinstance(dummy_env.action_space, Discrete)

    for label in ["", "guard", "move", "move_progress", "position"]:
        data_label = ("_" + label) if label else label

        runs_dataset = {
            k: game_model_(
                observation_space_size=dummy_env.observation_space.shape[0],
                action_space_size=int(dummy_env.action_space.n),
                **v, # type: ignore
            )[0]
            for k, v in runs_dataset_raw.items()
        }

        dfs = get_data_dataset(
            result_path=result_basename,
            runs=runs_dataset,
            observer_type=GameModelObserver,
            seeds=seeds,
            processes=min(processes, 6), # the PC gets way too hot if all CPUs are constantly running
            epochs=epochs,
            shuffle=shuffle,
            y=y,
        )

        if dfs is None:
            return

        # The residual model stopped sooner for some reason, so we cut the other runs at the same point as well
        max_idx = min(df["Idx"].max() for df in dfs.values())
        for name, df in dfs.items():
            dfs[name] = df[df["Idx"] <= max_idx].copy(deep=True)
        
        plot_data(
            dfs=dfs,
            title="",
            fig_path=f"{result_basename}_gm_dataset{data_label}",
            exp_factor=0.9,
            xlabel="Iteration",
            ylabel="Loss",
            run_name_mapping={
                "dataset_gm_residual":     "Residual",
                "dataset_gm_normal":       "Normal",
                "dataset_gm_differences":  "Differences",
            },
            attr_name=f"loss{data_label}",
            # yscale="log",
        )

        plot_data(
            dfs=dfs,
            title="",
            fig_path=f"{result_basename}_gm_dataset{data_label}_val",
            exp_factor=0.9,
            xlabel="Iteration",
            ylabel="Loss",
            run_name_mapping={
                "dataset_gm_residual":     "Residual",
                "dataset_gm_normal":       "Normal",
                "dataset_gm_differences":  "Differences",
            },
            attr_name=f"loss{data_label}_val",
            # yscale="log",
        )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)