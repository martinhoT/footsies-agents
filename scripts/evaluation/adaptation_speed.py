# %% Make sure we are running in the project's root

from os import chdir

chdir("/home/martinho/projects/footsies-agents")

# %% Imports

from os import path
from scripts.evaluation.utils import get_data_custom_loop, create_env

# %% Instantiate a blank agent

from models import to_

dummy_env, _  = create_env()

agent, _ = to_(
    observation_space_size=dummy_env.observation_space.shape[0],
    action_space_size=dummy_env.action_space.n,
)

# %% Check if all necessary runs have been made

from itertools import combinations
from copy import deepcopy
from opponents.curriculum import Idle, Backer, NSpammer, BSpammer, NSpecialSpammer, BSpecialSpammer, WhiffPunisher

agent_labels = ["blank", "idle", "backer", "n_spammer", "b_spammer", "n_special_spammer", "b_special_spammer", "whiff_punisher", "bot"]

agents = [
    (f"{training_opponent}_to_{evaluation_opponent}", deepcopy(agent), )
    for training_opponent, evaluation_opponent in combinations(opponent_labels)
]

get_data(
    data="win_rate",
    agents=agents,
    title="Win rate over the last 100 episodes against the in-game bot",
    fig_path=path.splitext(__file__)[0],
    seeds=1 # 10, # use less seeds for now
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
