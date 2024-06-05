from os import path
from scripts.evaluation.data_collectors import get_data, get_data_custom_loop
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import create_agent_the_one, quick_agent_args, quick_train_args
from scripts.evaluation.custom_loop import AgentCustomRun, WinRateObserver
from opponents.curriculum import WhiffPunisher
from gymnasium import Env
from functools import partial


def create_agent_the_one_ignore_hitstop(env: Env, ignore_hitstop: bool):
    return create_agent_the_one(env, {"one_decision_at_hitstop": ignore_hitstop})


def main(seeds: int | None = None, timesteps: int = int(2e6), processes: int = 12, y: bool = False):
    if seeds is None:
        seeds = 10
    
    result_path = path.splitext(__file__)[0]
    
    # The value is whether it is ignored!
    runs_raw = {
        "hitstop_keep": False,
        "hitstop_ignore": True,
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
        },
        ylim=(0, 1),
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

    # Prediction score of player model

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
        fig_path=result_path + "_score_opp",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Score",
        run_name_mapping={
            "hitstop_keep":     "Hitstop/blockstop kept",
            "hitstop_ignore":   "Hitstop/blockstop skipped",
        },
        ylim=(0, 1),
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
            "hitstop_keep":     "Hitstop/blockstop kept",
            "hitstop_ignore":   "Hitstop/blockstop skipped",
        }
    )

    # Against WhiffPunisher

    runs_whiff_raw = {
        "hitstop_keep_whiff": False,
        "hitstop_ignore_whiff": True,
    }

    whiff_punisher = WhiffPunisher()

    runs_whiff = {k: AgentCustomRun(
        agent=partial(create_agent_the_one_ignore_hitstop, ignore_hitstop=ignore_hitstop),
        opponent=whiff_punisher.act,
        skip_freeze=ignore_hitstop,
    ) for k, ignore_hitstop in runs_whiff_raw.items()}

    dfs = get_data_custom_loop(
        result_path=result_path + "_whiff",
        runs=runs_whiff,
        observer_type=WinRateObserver,
        seeds=seeds,
        timesteps=timesteps,
        processes=processes,
        y=y,
    )

    if dfs is None:
        return
    
    plot_data(
        dfs=dfs,
        title="",
        fig_path=result_path + "_whiff_wr",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "hitstop_keep_whiff": "Hitstop/blockstop kept",
            "hitstop_ignore_whiff": "Hitstop/blockstop skipped"
        },
        attr_name="win_rate",
        ylim=(0, 1),
    )


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
