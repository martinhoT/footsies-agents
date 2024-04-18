# %% Make sure we are running in the project's root

from os import chdir

chdir("/home/martinho/projects/footsies-agents")

# %% Imports

from os import path
from scripts.evaluation.utils import get_and_plot_data

# %% Check if all necessary runs have been made

agents = [
    ("sparse_reward", {}, {"dense_reward", False}, {"curriculum": True}),
    ("dense_reward", {}, {"dense_reward", True}, {"curriculum": True}),
]

get_and_plot_data(
    data="performancewin_rate_against_current_curriculum_opponent",
    agents=agents,
    title="Win rate over the last 100 episodes against the curriculum",
    fig_path=path.splitext(__file__)[0],
    seeds=10,
    timesteps=1000000,
    exp_factor=0.9,
    xlabel="Time step",
    ylabel="Win rate",
    run_name_mapping={
        "sparse_reward":    "Sparse reward",
        "dense_reward":     "Dense reward",
    }
)
