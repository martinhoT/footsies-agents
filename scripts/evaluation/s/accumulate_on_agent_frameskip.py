from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_train_args

def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 12, y: bool = False):
    result_path = path.splitext(__file__)[0]
    
    runs_raw = {
        "yes_agent_frameskip": {"accumulate_at_frameskip": True},
        "no_agent_frameskip": {"accumulate_at_frameskip": False}
    }

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k, kwargs=v),
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
        fig_path=result_path + "_wr",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "yes_agent_frameskip":  "Decision skip",
            "no_agent_frameskip":   "No decision skip",
        },
    )

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
        title="Policy entropy against the in-game AI",
        fig_path=result_path + "_policy_ent",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "yes_agent_frameskip":  "Decision skip",
            "no_agent_frameskip":   "No decision skip",
        },
    )

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
        title="Opponent model loss against the in-game AI",
        fig_path=result_path + "_opp_loss",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Loss",
        run_name_mapping={
            "yes_agent_frameskip":  "Decision skip",
            "no_agent_frameskip":   "No decision skip",
        },
    )

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
        title="Opponent model's prediction score against the in-game AI",
        fig_path=result_path + "_opp_score",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Score",
        run_name_mapping={
            "yes_agent_frameskip":  "Decision skip",
            "no_agent_frameskip":   "No decision skip",
        },
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
