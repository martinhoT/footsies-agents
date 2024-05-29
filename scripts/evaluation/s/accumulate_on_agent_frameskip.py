from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_train_args

def main(seeds: int | None = None, timesteps: int = int(1e6), processes: int = 12, y: bool = False):
    if seeds is None:
        seeds = 10
    
    result_path = path.splitext(__file__)[0]
    
    runs_raw = {
        "yes_agent_frameskip": {"accumulate_at_frameskip": True},
        "no_agent_frameskip": {"accumulate_at_frameskip": False}
    }

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k, kwargs=v),
        timesteps=timesteps,
    ) for k, v in runs_raw.items()}

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
        title="",
        fig_path=result_path + "_wr",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "yes_agent_frameskip":  "Decision skip",
            "no_agent_frameskip":   "No decision skip",
        },
        ylim=(0, 1),
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
            "yes_agent_frameskip":  "Decision skip",
            "no_agent_frameskip":   "No decision skip",
        },
        ylim=(0, None),
    )

    # Advantage

    dfs = get_data(
        data="learningdelta",
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
        fig_path=result_path + "_delta",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Advantage",
        run_name_mapping={
            "yes_agent_frameskip":  "Decision skip",
            "no_agent_frameskip":   "No decision skip",
        },
    )

    # Delta

    dfs = get_data(
        data="learningqlearner_error",
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
        fig_path=result_path + "_qerror",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="TD error",
        run_name_mapping={
            "yes_agent_frameskip":  "Decision skip",
            "no_agent_frameskip":   "No decision skip",
        },
    )


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
