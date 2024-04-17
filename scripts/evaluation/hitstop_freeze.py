# %% [markdown]

# **Setup**: main agent with and without hitstop freeze
# - `eval_hitstop_keep`
# - `eval_hitstop_ignore`

# %% Make sure we are running in the project's root

from os import chdir

chdir("/home/martinho/projects/footsies-agents")

# %% Imports

from os import path
from scripts.evaluation.utils import get_data

# %% Check if all necessary runs have been made

neededs = [
    ("hitstop_keep", {}, {}, {"skip_freeze": False}),
    ("hitstop_ignore", {}, {}, {"skip_freeze": True}),
]

dfs = get_data("win_rate", neededs, seeds=10, timesteps=2500000)

# %% Plot the data

import matplotlib.pyplot as plt

result_path, _ = path.splitext(__file__)

# Smooth the values (make exponential moving average) and plot them
for name, df in dfs:
    plt.fill_between(df.Idx, df.ValMean - df.ValStd, df.ValMean + df.ValStd)
    df.plot.line(x="Idx", y="ValueMean")

plt.legend([name for name, _, _ in neededs])
plt.title("Win rate over the last 100 episodes against the in-game bot")
plt.xlabel("Episode")
plt.ylabel("Win rate")
plt.savefig(result_path)
