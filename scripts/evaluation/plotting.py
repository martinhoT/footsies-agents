import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import isnan
from opponents.curriculum import CurriculumManager


def plot_data(
    dfs: dict[str, pd.DataFrame],
    *,
    title: str = "",
    fig_path: str | None = None,
    exp_factor: float = 0.9,
    xlabel: str | None = None,
    ylabel: str | None = None,
    yscale: str = "linear",
    ylim: tuple[float | None, float | None] | None = None,
    ylim_truncate: bool = True, # only set ylim if the graph, by default, goes beyond those limits
    run_name_mapping: dict[str, str] | None = None,
    attr_name: str = "Val"
):
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
        if df[c_se].isna().all():
            continue

        plt.fill_between(df.Idx, df[c_me] - df[c_se], df[c_me] + df[c_se], alpha=0.2)

    if run_name_mapping is not None:
        plt.legend([run_name_mapping[name] for name in dfs.keys()])
    plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if ylim is not None:
        bottom, top = plt.ylim()
        new_bottom, new_top = ylim
        if new_bottom is not None and not (ylim_truncate and bottom >= new_bottom):
            bottom = new_bottom
        if new_top is not None and not (ylim_truncate and top <= new_top):
            top = new_top
        plt.ylim(bottom, top)

    plt.yscale(yscale)
    plt.tight_layout()

    if fig_path is not None:
        plt.savefig(fig_path)
        plt.clf()


def plot_add_curriculum_transitions(dfs_transitions: dict[str, pd.DataFrame], seeds: int, fig_path: str | None):
    n = len(dfs_transitions)
    linewidth = 4

    ax = plt.gca()
    for i, (_, df) in enumerate(dfs_transitions.items()):
        linestyle = (i * linewidth, (n, n * linewidth))

        transition_idxs = []
        for j in range(seeds):
            grouped = df.groupby(f"Val{j}")["Idx"].apply(list)
            if 1 not in grouped:
                raise ValueError(f"no transitions were detected for seed {j} of run {i}")
            transition_idxs.append(grouped[1])

        transitions = [sum(v) / len(v) for v in map(lambda vs: [v for v in vs if not isnan(v)], zip(*transition_idxs))]
        ax.vlines(transitions, 0, 1, linestyles=linestyle, colors=f"C{i}")

    if fig_path is not None:
        plt.savefig(fig_path)
        plt.clf()


def plot_curriculum(dfs: dict[str, pd.DataFrame], dfs_transitions: dict[str, pd.DataFrame], seeds: int, *, title: str = "", fig_path: str | None = None, exp_factor: float = 0.9, ylabel: str | None = None, yscale: str = "linear", run_name_mapping: dict[str, str] | None = None, attr_name: str = "Val", ignore_dummy_opps: bool = True):
    # The number of opponents that the curriculum has by default,
    # which is what we assume the runs used.
    n_curriculum_opponents = CurriculumManager().remaining
    
    # This like the input dfs but instead we have one for each curriculum opponent.
    split_dfs: list[dict[str, pd.DataFrame]] = []

    seed_columns = [f"{attr_name}{i}" for i in range(seeds)]
    for opp in range(n_curriculum_opponents):
        if ignore_dummy_opps and opp < 2:
            continue

        opp_split_dfs = {k: pd.DataFrame(columns=["Idx"]) for k in dfs}

        for name, opp_df, full_df, full_df_transitions in zip(opp_split_dfs.keys(), opp_split_dfs.values(), dfs.values(), dfs_transitions.values()):
            for seed_col in seed_columns:
                opponent_completed_idxs = full_df_transitions.index[full_df_transitions[seed_col] == 1.0]

                # Did not get to this opponent...
                if opp >= len(opponent_completed_idxs):
                    opp_df[seed_col] = np.nan
                    continue

                # Join on index
                prev_idx = (opponent_completed_idxs[opp - 1] + 1) if opp > 0 else 0
                curr_idx = (opponent_completed_idxs[opp] + 1)
                opp_data = full_df.iloc[prev_idx:curr_idx][seed_col].reset_index(drop=True)
                opp_df = opp_df.join(opp_data, how="outer", on=None)

            opp_df["Idx"] = range(len(opp_df))
            opp_df[f"{attr_name}Mean"] = opp_df[seed_columns].mean(axis=1)
            opp_df[f"{attr_name}Std"] = opp_df[seed_columns].std(axis=1)
            # Because there is no inplace join
            opp_split_dfs[name] = opp_df

        split_dfs.append(opp_split_dfs)
    
    for opp, d in enumerate(split_dfs):
        plot_data(
            dfs=d,
            title=title,
            fig_path=f"{fig_path}_OPP{opp}" if fig_path is not None else None,
            exp_factor=exp_factor,
            xlabel="Episode",
            ylabel=ylabel,
            yscale=yscale,
            run_name_mapping=run_name_mapping,
            attr_name=attr_name,
        )

    return split_dfs
