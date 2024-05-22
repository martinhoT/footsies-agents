from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_train_args

def main(seeds: int | None = None, timesteps: int = int(1e6), processes: int = 12, y: bool = False):
    if seeds is None:
        seeds = 1
    
    runs_raw = {
        "discount_1_0": {"critic_discount": 1.0, "policy_cumulative_discount": False},
        "discount_0_99": {"critic_discount": 0.99, "policy_cumulative_discount": False},
        "discount_0_9": {"critic_discount": 0.9, "policy_cumulative_discount": False},
        "discount_1_0_correct": {"critic_discount": 1.0, "policy_cumulative_discount": True},
        "discount_0_99_correct": {"critic_discount": 0.99, "policy_cumulative_discount": True},
        "discount_0_9_correct": {"critic_discount": 0.9, "policy_cumulative_discount": True},
    }

    from scripts.evaluation.utils import quick_env_args

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k, kwargs=v),
        env_args=quick_env_args(
            kwargs={
                "log_file": "/home/martinho/projects/footsies-agents/eval1.log",
                "log_file_overwrite": True,
            },
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
        title="",
        fig_path=path.splitext(__file__)[0],
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "discount_1_0":             "$\\gamma = 1.0$",
            "discount_0_999":           "$\\gamma = 0.999$",
            "discount_0_99":            "$\\gamma = 0.99$",
            "discount_0_9":             "$\\gamma = 0.9$",
            "discount_1_0_correct":     "$\\gamma = 1.0$ (correct)",
            "discount_0_999_correct":   "$\\gamma = 0.999$ (correct)",
            "discount_0_99_correct":    "$\\gamma = 0.99$ (correct)",
            "discount_0_9_correct":     "$\\gamma = 0.9$ (correct)",
        }
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
