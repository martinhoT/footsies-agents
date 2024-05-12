import pandas as pd
import matplotlib.pyplot as plt
from math import isnan


def plot_data(dfs: dict[str, pd.DataFrame], *, title: str, fig_path: str | None, exp_factor: float = 0.9, xlabel: str | None = None, ylabel: str | None = None, yscale: str = "linear", run_name_mapping: dict[str, str] | None = None, attr_name: str = "Val"):
    # Compact columns names (mean, std, mean exp, std exp)
    c_m = f"{attr_name}Mean"
    c_s = f"{attr_name}Std"
    c_me = f"{attr_name}MeanExp"
    c_se = f"{attr_name}StdExp"
    
    # Smooth the values (make exponential moving average) and plot them
    alpha = 1 - exp_factor
    for df in dfs.values():
        df[c_me] = df[c_m].ewm(alpha=alpha).mean()
        df[c_se] = df[c_s].ewm(alpha=alpha).mean()
        plt.plot(df.Idx, df[c_me])

    for df in dfs.values():
        plt.fill_between(df.Idx, df[c_me] - df[c_se], df[c_me] + df[c_se], alpha=0.2)

    if run_name_mapping is not None:
        plt.legend([run_name_mapping[name] for name in dfs.keys()])
    plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.yscale(yscale)

    if fig_path is not None:
        plt.savefig(fig_path)
        plt.clf()


def plot_add_curriculum_transitions(dfs_transitions: dict[str, pd.DataFrame], seeds: int, fig_path: str | None):
    n = len(dfs_transitions)
    linewidth = 4

    ax = plt.gca()
    for i, (_, df) in enumerate(dfs_transitions.items()):
        linestyle = (i * linewidth, (n, (n - 1) * linewidth))

        transition_idxs = [df.groupby(f"Val{j}")["Idx"].apply(list)[1] for j in range(seeds)]
        transitions = [sum(v) / len(v) for v in map(lambda vs: [v for v in vs if not isnan(v)], zip(*transition_idxs))]
        ax.vlines(transitions, 0, 1, linestyles=linestyle, colors=f"C{i}")

    if fig_path is not None:
        plt.savefig(fig_path)
        plt.clf()
