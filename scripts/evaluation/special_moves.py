# %% Make sure we are running in the project's root

from os import chdir

chdir("/home/martinho/projects/footsies-agents")

# %% Imports

from os import path
from scripts.evaluation.utils import get_and_plot_data

# %% Check if all necessary runs have been made

agents = [
    ("no_specials", {"remove_special_moves": True}, {}, {}),
    ("yes_specials", {"remove_special_moves": False}, {}, {}),
]

get_and_plot_data(
    data="win_rate",
    agents=agents,
    title="Win rate over the last 100 episodes against the in-game bot",
    fig_path=path.splitext(__file__)[0],
    seeds=10,
    timesteps=1000000,
    exp_factor=0.9,
    xlabel="Time step",
    ylabel="Win rate",
    run_name_mapping={
        "no_specials":  "Without special moves",
        "yes_specials": "With special moves",
    }
)
