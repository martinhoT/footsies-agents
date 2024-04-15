# %% [markdown]

# Setup: main agent with various
# - 

# %% Make sure we are running in the project's root

from os import chdir
chdir("/home/martinho/projects/footsies-agents")

# %% Imports

from os import path
from main import load_agent_parameters

# %% Check if all necessary runs have been made

agent_ref = "f_opp_recurrent_no_mask"
baseline_ppo = "sb3_ppo"
baseline_a2c = "sb3_a2c"
baseline_dqn = "sb3_dqn"

neededs = [agent_ref, baseline_ppo, baseline_a2c, baseline_dqn]

all_good = True
for needed in neededs:
    data_path = path.join("saved", needed)
    if not path.exists(data_path):
        all_good = False
        print(f"Please run the runner script for '{needed}'")

if not all_good:
    exit(1)

# %% Check if those runs had the correct arguments (they need to be similar)

agent_ref_parameters = load_agent_parameters(agent_ref)
baseline_ppo_parameters = load_agent_parameters(baseline_ppo)
baseline_a2c_parameters = load_agent_parameters(baseline_a2c)
baseline_dqn_parameters = load_agent_parameters(baseline_dqn)

correct_any_parameters = {
    "learning_rate": 1e-4,
    "gamma": 0.9,
}
correct_a2c_ppo_parameters = {
    "normalize_advantage": False,
    "ent_coef": 0.04, # don't know if this is necessarily equivalent to ours!!!
    "max_grad_norm": 0.5,
    "use_sde": False,
    "gae_lambda": "|DEFAULT|" # note that in order to be equivalent to my method it should be 0
}
correct_a2c_parameters = {
    "use_rms_prop": False, # to match PPO in using Adam, even though I'm not using it
}
correct_ppo_parameters = {
    "n_epochs": 10,
}
correct_dqn_ppo_parameters = {
    "batch_size": 64,
}
correct_dqn_parameters = {
    # just use the defaults, the method is barely comparable
}

for parameter, value in correct_any_parameters.items():
    if value == "|DEFAULT|":
        continue
    assert baseline_ppo_parameters[parameter] == value
    assert baseline_a2c_parameters[parameter] == value
    assert baseline_dqn_parameters[parameter] == value

for parameter, value in correct_a2c_ppo_parameters.items():
    if value == "|DEFAULT|":
        continue
    assert baseline_ppo_parameters[parameter] == value
    assert baseline_a2c_parameters[parameter] == value

for parameter, value in correct_a2c_parameters.items():
    if value == "|DEFAULT|":
        continue
    assert baseline_a2c_parameters[parameter] == value

for parameter, value in correct_ppo_parameters.items():
    if value == "|DEFAULT|":
        continue
    assert baseline_ppo_parameters[parameter] == value

for parameter, value in correct_dqn_ppo_parameters.items():
    if value == "|DEFAULT|":
        continue
    assert baseline_ppo_parameters[parameter] == value
    assert baseline_dqn_parameters[parameter] == value

for parameter, value in correct_dqn_parameters.items():
    if value == "|DEFAULT|":
        continue
    assert baseline_dqn_parameters[parameter] == value

assert agent_ref_parameters["critic_discount"] == correct_any_parameters["gamma"]
assert agent_ref_parameters["actor_lr"] == correct_any_parameters["learning_rate"]
assert agent_ref_parameters["critic_lr"] == correct_any_parameters["learning_rate"]
assert agent_ref_parameters["actor_entropy_coef"] == correct_a2c_ppo_parameters["ent_coef"]
assert agent_ref_parameters["actor_gradient_clipping"] == correct_a2c_ppo_parameters["max_grad_norm"]

# %% Check if, from all runs, the data was exported to CSV, and load it

import pandas as pd

all_good = True
datas: list[pd.DataFrame] = []
for needed in neededs:
    data_path = path.join("saved", needed, "win_rate.csv")
    if not path.exists(data_path):
        all_good = False
        print(f"Please save the data regarding win rates for '{needed}'")
    
    elif all_good:
        data = pd.read_csv(data_path)
        # Only read a portion of the data
        data = data[data["Step"] <= 5e6]
        datas.append(data)

if not all_good:
    exit(1)

# %% Plot the data

import matplotlib.pyplot as plt

result_path, _ = path.splitext(__file__)

# Smooth the values (make exponential moving average) and plot them
for data in datas:
    data["ValueExp"] = data["Value"].ewm(alpha=0.1).mean()
    data.plot.line(x="Step", y="ValueExp")

plt.legend(["Ours", "PPO", "A2C", "DQN"])
plt.title("Win rate over the last 100 episodes against the in-game bot")
plt.xlabel("Episode")
plt.ylabel("Win rate")
plt.savefig(result_path)
