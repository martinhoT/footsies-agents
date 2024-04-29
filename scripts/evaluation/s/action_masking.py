from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_train_args

def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 4, y: bool = False):
    result_path = path.splitext(__file__)[0]
    
    runs_raw = {
        "action_mask_off": {"action_masking": False},
        "action_mask_on": {"action_masking": True},
    }
    
    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k, model="to", kwargs=v),
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
        title="Win rate over the last 100 episodes against the in-game bot",
        fig_path=result_path + "_wr",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "action_mask_off":  "No action masking",
            "action_mask_on":   "Action masking",
        },
    )

    dfs = get_data(
        data="learningpolicy_entropy",
        runs=runs,
        seeds=seeds,
        processes=processes,
    )

    if dfs is None:
        return

    plot_data(
        dfs=dfs,
        title="Policy entropy while playing against the in-game bot",
        fig_path=result_path + "_ent",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Entropy",
        run_name_mapping={
            "action_mask_off":  "No action masking",
            "action_mask_on":   "Action masking",
        },
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
