# %% Make sure we are running in the project's root

from os import chdir

chdir("/home/martinho/projects/footsies-agents")

# %% Imports

from os import path
from scripts.evaluation.utils import get_and_plot_data

# %% Check if all necessary runs have been made

agents = [
    ("discount_1_0", {"critic_discount": 1.0, "policy_cumulative_discount": False}, {}, {}),
    # ("discount_0_999", {"critic_discount": 0.999, "policy_cumulative_discount": False}, {}, {}),
    ("discount_0_99", {"critic_discount": 0.99, "policy_cumulative_discount": False}, {}, {}),
    ("discount_0_9", {"critic_discount": 0.9, "policy_cumulative_discount": False}, {}, {}),
    ("discount_1_0_correct", {"critic_discount": 1.0, "policy_cumulative_discount": True}, {}, {}),
    # ("discount_0_999_correct", {"critic_discount": 0.999, "policy_cumulative_discount": True}, {}, {}),
    ("discount_0_99_correct", {"critic_discount": 0.99, "policy_cumulative_discount": True}, {}, {}),
    ("discount_0_9_correct", {"critic_discount": 0.9, "policy_cumulative_discount": True}, {}, {}),
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
        "discount_1_0":             "$\gamma = 1.0$",
        "discount_0_999":           "$\gamma = 0.999$",
        "discount_0_99":            "$\gamma = 0.99$",
        "discount_0_9":             "$\gamma = 0.9$",
        "discount_1_0_correct":     "$\gamma = 1.0$ (correct)",
        "discount_0_999_correct":   "$\gamma = 0.999$ (correct)",
        "discount_0_99_correct":    "$\gamma = 0.99$ (correct)",
        "discount_0_9_correct":     "$\gamma = 0.9$ (correct)",
    }
)
