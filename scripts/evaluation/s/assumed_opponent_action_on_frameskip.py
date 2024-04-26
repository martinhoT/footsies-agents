from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_env_args, quick_train_args

def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 4):
    runs_raw = {
        "assumed_opponent_action_on_frameskip_last": {"footsies_wrapper_simple": (True, True, "last", "last")},
        "assumed_opponent_action_on_frameskip_none": {"footsies_wrapper_simple": (True, True, "last", "none")},
        "assumed_opponent_action_on_frameskip_stand": {"footsies_wrapper_simple": (True, True, "last", "stand")},
    }
    
    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k),
        env_args=quick_env_args(**v),
        timesteps=timesteps,
    ) for k, v in runs_raw.items()}

    dfs = get_data(
        data="win_rate",
        runs=runs,
        seeds=seeds,
        processes=processes,
    )

    if dfs is None:
        return

    plot_data(
        dfs=dfs,
        title="Win rate over the last 100 episodes against the in-game bot",
        fig_path=path.splitext(__file__)[0],
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "assumed_opponent_action_on_frameskip_last":    "Last",
            "assumed_opponent_action_on_frameskip_none":    "None",
            "assumed_opponent_action_on_frameskip_stand":   "No-op",
        },
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
