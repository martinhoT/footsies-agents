# %% [markdown]

# **Setup**: main agent with various discount factors
# - `eval_discount_1_0`
# - `eval_discount_0_999`
# - `eval_discount_0_99`
# - `eval_discount_0_9`
# - `eval_discount_1_0_correct`
# - `eval_discount_0_999_correct`
# - `eval_discount_0_99_correct`
# - `eval_discount_0_9_correct`

# %% Make sure we are running in the project's root

from os import chdir

chdir("/home/martinho/projects/footsies-agents")

# %% Imports

from os import path
from scripts.evaluation.utils import get_data

# %% Check if all necessary runs have been made

neededs = [
    ("discount_1_0", {"critic_discount": 1.0, "policy_cumulative_discount": False}, {}),
    ("discount_0_999", {"critic_discount": 0.999, "policy_cumulative_discount": False}, {}),
    ("discount_0_99", {"critic_discount": 0.99, "policy_cumulative_discount": False}, {}),
    ("discount_0_9", {"critic_discount": 0.9, "policy_cumulative_discount": False}, {}),
    ("discount_1_0_correct", {"critic_discount": 1.0, "policy_cumulative_discount": True}, {}),
    ("discount_0_999_correct", {"critic_discount": 0.999, "policy_cumulative_discount": True}, {}),
    ("discount_0_99_correct", {"critic_discount": 0.99, "policy_cumulative_discount": True}, {}),
    ("discount_0_9_correct", {"critic_discount": 0.9, "policy_cumulative_discount": True}, {}),
]

dfs = get_data("win_rate", neededs, seeds=10, timesteps=1000000)

# %% Plot the data

import matplotlib.pyplot as plt

result_path, _ = path.splitext(__file__)

# Smooth the values (make exponential moving average) and plot them
for name, df in dfs:
    # df["ValExp"] = df["ValMean"].ewm(alpha=0.1).mean()
    plt.fill_between(df.Idx, df.ValMean - df.ValStd, df.ValMean + df.ValStd)
    df.plot.line(x="Step", y="ValueExp")

plt.legend([name for name, _, _ in neededs])
plt.title("Win rate over the last 100 episodes against the in-game bot")
plt.xlabel("Episode")
plt.ylabel("Win rate")
plt.savefig(result_path)
