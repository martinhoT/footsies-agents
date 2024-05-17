from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_train_args

def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 12, y: bool = False):
    result_path = path.splitext(__file__)[0]
    
    runs_raw = {
        "hitstop_keep": True,
        "hitstop_ignore": False,
    }

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k,
            kwargs={
                "one_decision_at_hitstop": ignore_hitstop,
                "use_game_model": True,
                "learn": "all",
            },
        ),
        timesteps=timesteps,
        skip_freeze=ignore_hitstop,
    ) for k, ignore_hitstop in runs_raw.items()}

    # Win rate of normal agent

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
            "hitstop_keep":     "Hitstop/blockstop kept",
            "hitstop_ignore":   "Hitstop/blockstop skipped",
        }
    )

    # Loss of player model

    dfs = get_data(
        data="learningloss_of_p2s_model",
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
        fig_path=result_path + "_loss_opp",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Loss",
        run_name_mapping={
            "hitstop_keep":     "Hitstop/blockstop kept",
            "hitstop_ignore":   "Hitstop/blockstop skipped",
        }
    )

    # Game model loss

    dfs = get_data(
        data="learninggame_model_loss",
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
        fig_path=result_path + "_loss_gm",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Loss",
        run_name_mapping={
            "hitstop_keep":     "Hitstop/blockstop kept",
            "hitstop_ignore":   "Hitstop/blockstop skipped",
        }
    )


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
