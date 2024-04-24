from os import path
from scripts.evaluation.utils import get_and_plot_data

def main(seeds: int):
    agents = [
        ("hitstop_keep", {}, {}, {"skip_freeze": False}),
        ("hitstop_ignore", {}, {}, {"skip_freeze": True}),
    ]

    get_and_plot_data(
        data="win_rate",
        agents=agents,
        title="Win rate over the last 100 episodes against the in-game bot",
        fig_path=path.splitext(__file__)[0],
        seeds=seeds,
        timesteps=1000000,
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "hitstop_keep":     "Hitstop kept",
            "hitstop_ignore":   "Hitstop skipped",
        }
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)