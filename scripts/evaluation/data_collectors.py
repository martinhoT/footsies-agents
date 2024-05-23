import os
import pandas as pd
import multiprocessing as mp
import shutil
import json
import inspect
from models import to_
from datetime import datetime
from typing import Sequence, Mapping, Any
from agents.base import FootsiesAgentBase
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.envs.exceptions import FootsiesGameClosedError
from agents.game_model.agent import GameModelAgent
from agents.mimic.agent import MimicAgent
from args import MainArgs
from os import path
from scripts.evaluation.custom_loop import Observer, custom_loop, dataset_run, AgentCustomRun
from functools import partial, reduce
from main import main, save_agent
from dataclasses import replace, asdict


# Wrapper on custom_loop
def custom_loop_df(
    save_path: str,
    run: AgentCustomRun,
    label: str,
    id_: int,
    observer_type: type[Observer],
    seed: int | None = 0,
    timesteps: int = int(1e6),
) -> pd.DataFrame:
    
    if os.path.exists(save_path):
        return pd.read_csv(save_path)

    done = False
    while not done:
        try:
            observer, agent = custom_loop(run, label, id_, observer_type, seed, timesteps)
            done = True
        except FootsiesGameClosedError:
            print(f"Game closed unexpectedly, will try running this run ({id_}) again")
    
    df = observer.df(str(seed))
    df.to_csv(save_path, header=df.columns.tolist(), index=False)
    
    # Save the agent as well
    basename, _ = os.path.splitext(os.path.basename(save_path))
    save_agent(agent, "eval_" + basename)
    
    return df


# Wrapper on dataset_run
def dataset_run_df(
    save_path: str,
    agent: MimicAgent | GameModelAgent,
    label: str,
    observer_type: type[Observer],
    seed: int | None = 0,
    epochs: int = 100,
    shuffle: bool = True,
) -> pd.DataFrame:
    
    if os.path.exists(save_path):
        return pd.read_csv(save_path)

    observer = dataset_run(agent, label, observer_type, seed, epochs, shuffle)
    df = observer.df(str(seed))
    df.to_csv(save_path, index=False)

    # Save the agent as well
    basename, _ = os.path.splitext(os.path.basename(save_path))
    save_agent(agent, "eval_" + basename)

    return df


# Wrapper on main
def main_ensured(args: MainArgs):
    done = False
    while not done:
        try:
            main(args)
            done = True
        except FootsiesGameClosedError:
            print(f"Game closed unexpectedly, will try running this run ({args.agent.name}) again")


def get_data_custom_loop(result_path: str, runs: dict[str, AgentCustomRun], observer_type: type[Observer], seeds: int = 10, timesteps: int = int(1e6), processes: int = 4, y: bool = False) -> dict[str, pd.DataFrame] | None:
    # Halve the number of processes since there are technically going to be two processes per run: the agent and the game
    processes //= 2

    dfs: dict[str, pd.DataFrame] = {}

    # `None` means all seeds are missing, and so a new dataframe has to be created
    missing: dict[str, list[int] | None] = {}
    for run_name in runs:
        df_path = f"{result_path}_{run_name}.csv"
        if path.exists(df_path):
            df = pd.read_csv(df_path)
            dfs[run_name] = df
            missing_seeds = sorted({seed for seed in range(seeds) for attribute in observer_type.attributes() if f"{attribute}{seed}" not in df.columns})
            if missing_seeds:
                missing[run_name] = missing_seeds

        else:
            missing[run_name] = None

    if missing:
        if not y:
            names_list = [f"- {run_name} (seeds: {seeds})" for run_name, seeds in missing.items()]
            print("The following runs are missing:", *names_list, sep="\n")
            ans = input("Do you want to run them now? [y/N] ")
            if ans.upper() != "Y":
                return None

        # Collect the data

        with mp.get_context("spawn").Pool(processes=processes) as pool:
            custom_loop_partial = partial(custom_loop_df, timesteps=timesteps)

            args: list[tuple[str, AgentCustomRun, str, int, type[Observer], int]] = []
            for i, (run_name, missing_seeds) in enumerate(missing.items()):
                run_args = runs[run_name]

                if missing_seeds is None:
                    missing_seeds = list(range(seeds))

                for seed in missing_seeds:
                    save_path = f"{result_path}_{run_name}_S{seed}_O.csv"
                    id_ = i * seeds + seed
                    a = (save_path, run_args, run_name, id_, observer_type, seed)
                    args.append(a)

            dfs_flat: list[pd.DataFrame] = pool.starmap(custom_loop_partial, args)
        
        # Create/update the dataframes with the observer data

        i = 0
        for run_name, missing_seeds in missing.items():
            # If None, then we want to consider all seeds
            if missing_seeds is None:
                missing_seeds = list(range(seeds))

            # Get the dataframes pertaining to this run only
            missing_seeds_n = len(missing_seeds)
            dfs_to_merge = dfs_flat[i:i+missing_seeds_n]
            i += missing_seeds_n

            # If there is already a dataframe, then merge the new ones with it
            if run_name in dfs:
                dfs_to_merge.append(dfs[run_name])
            
            # Perform the merge...
            merge_method = partial(pd.merge, how="outer", on="Idx")
            merged_df = reduce(merge_method, dfs_to_merge)

            # ... and store
            dfs[run_name] = merged_df

        # Save the merged data for posterity. At this point the individual observer data is useless

        for run_name in missing:
            df = dfs[run_name]
            df.to_csv(f"{result_path}_{run_name}.csv", index=False)
    
    # Calculate aggregations according to the seeds

    for _, df in dfs.items():
        for attribute in observer_type.attributes():
            attribute_seed_columns = [f"{attribute}{seed}" for seed in range(seeds)]
            df[f"{attribute}Mean"] = df[attribute_seed_columns].mean(axis=1)
            df[f"{attribute}Std"] = df[attribute_seed_columns].std(axis=1)

    return dfs


def all_eval_trained() -> dict[str, dict[str, Any]]:
    res: dict[str, dict[str, Any]] = {}

    for eval_run in os.listdir("saved"):
        if not eval_run.startswith("eval_"):
            continue

        eval_path = os.path.join("saved", eval_run)
        training_args_files = [f for f in os.listdir(eval_path) if f.startswith("training_args_")]
        training_args_files.sort(key=lambda f: datetime.strptime(f[14:-5], "%d-%m-%y_%H-%M-%S"), reverse=True)

        # It's possible that there are no training args files, such as when the agent was run in a custom loop
        if not training_args_files:
            continue

        most_recent_training_args = training_args_files[0]

        most_recent_training_args_path = os.path.join(eval_path, most_recent_training_args)
        with open(most_recent_training_args_path, "rt") as f:
            res[eval_run] = json.load(f)

    return res


def is_default(signature: inspect.Signature, argument: str, value: Any) -> bool:
    return signature.parameters[argument].default == value


def get_similar_agent_run(args: MainArgs) -> str | None:
    if args.agent.model != "to":
        raise ValueError("this method only supports the 'to' model")

    tosig = inspect.signature(to_)
    args_dict = asdict(args)

    for run, run_args in all_eval_trained().items():
        # We only want the_one runs
        if run_args["agent"]["model"] != "to":
            continue

        agent_args: dict[str, Any] = run_args["agent"]["kwargs"]

        # Check if the agent arguments are essentially the same.
        # Basically, they either need to be equal, or the defined value needs to be the default.
        all_keys = agent_args.keys() | args.agent.kwargs.keys()
        all_match = True
        for key in all_keys:
            if key in args.agent.kwargs:
                this_value = args.agent.kwargs[key]

                if key in agent_args:
                    other_value = agent_args[key]

                    if this_value == other_value:
                        continue
                
                else:
                    if is_default(tosig, key, this_value):
                        continue
                        
            else:
                other_value = agent_args[key]

                if is_default(tosig, key, other_value):
                    continue

            all_match = False
            break

        if not all_match:
            continue

        # Ignore all other arguments that don't matter
        run_args["agent"]["kwargs"] = args_dict["agent"]["kwargs"]
        run_args["agent"]["name_"] = args_dict["agent"]["name_"]
        run_args["env"]["kwargs"].pop("game_port")
        run_args["env"]["kwargs"].pop("opponent_port")
        run_args["env"]["kwargs"].pop("remote_control_port")
        run_args["progress_bar_kwargs"] = args_dict["progress_bar_kwargs"]

        # Now, check if all arguments are the same
        if run_args == args_dict:
            return run

    return None


def get_data(data: str, runs: dict[str, MainArgs], seeds: int = 10, processes: int = 4, y: bool = False, data_cols: Sequence[int] = (0, 1), pre_trained: str | None = None) -> dict[str, pd.DataFrame] | None:
    # Halve the number of processes since there are technically going to be two processes per run: the agent and the game
    processes //= 2
    
    missing: dict[str, list[tuple[str, int]]] = {}
    for run_name, main_args in runs.items():
        for seed in range(seeds):
            run_fullname = f"eval_{run_name}_S{seed}"
            data_path = path.join("runs", run_fullname)
            if not path.exists(data_path):
                # Try checking if there is another run that already does the same thing.
                # If so, copy the results.
                similar_run = get_similar_agent_run(main_args)
                if similar_run is not None:
                    print(f"Found a similar run for '{run_fullname}': '{similar_run}', will copy its results")
                    similar_data_path = path.join("runs", similar_run)
                    shutil.copytree(similar_data_path, data_path)
                    continue

                missing.setdefault(run_name, []).append((run_fullname, seed))

    if missing:
        if not y:
            names_list = [f"- {run_name} (seeds: {[seed for _, seed in missing_seeded_runs]})" for run_name, missing_seeded_runs in missing.items()]
            print("The following runs are missing:", *names_list, sep="\n")
            ans = input("Do you want to run them now? [y/N] ")
            if ans.upper() != "Y":
                return None
        
        # Try to avoid oversubscription, since each agent will have at least 2 processes running: itself, and the game
        with mp.get_context("spawn").Pool(processes=processes) as pool:
            args: list[MainArgs] = []
            for i, (run_name, missing_seeded_runs) in enumerate(missing.items()):
                run_args = runs[run_name]
                for run_fullname, seed in missing_seeded_runs:
                    # Update env args with specific ports for each run
                    port_start = 11000 + i*100 + seed*10
                    ports = FootsiesEnv.find_ports(port_start, stop=port_start + 10)
                    env_args_modified = replace(run_args.env, kwargs=run_args.env.kwargs | ports)

                    # Update agent args with specific name
                    agent_args_modified = replace(run_args.agent, name_=run_fullname)

                    # Update misc args with pre-training info
                    if pre_trained is not None:
                        misc_args_modified = replace(run_args.misc, load=True)

                        # Make sure there is a copy to load from
                        pre_trained_path = os.path.join("saved", pre_trained)
                        if not os.path.exists(pre_trained_path):
                            raise ValueError("The requested pre-trained agent doesn't exist")
                        
                        prepared_pre_trained_path = os.path.join("saved", run_fullname)
                        if os.path.exists(prepared_pre_trained_path):
                            timestamp = datetime.now().isoformat()
                            print(f"An agent at {prepared_pre_trained_path} already exists, sending it to the trash ({timestamp})")
                            home_path = os.environ["HOME"]
                            trash_path = os.path.join(home_path, ".local", "share", "Trash", "files", run_fullname + f"_{timestamp}")
                            os.rename(prepared_pre_trained_path, trash_path)
                        
                        shutil.copytree(pre_trained_path, prepared_pre_trained_path)
                        print(f"Copied pre-trained agent '{pre_trained_path}' to the path '{prepared_pre_trained_path}'")

                    else:
                        misc_args_modified = run_args.misc

                    # Update main args with everything + seed
                    run_args_modified = replace(run_args,
                        env=env_args_modified,
                        agent=agent_args_modified,
                        misc=misc_args_modified,
                        seed=seed,
                        raise_game_closed=True,
                    )
                    
                    args.append(run_args_modified)

            pool.map(main_ensured, args)
    
    dfs: dict[str, pd.DataFrame] = {}
    for run_name in runs:
        ds = [
            pd.read_csv(
                path.join("runs", f"eval_{run_name}_S{seed}", f"{data}.csv"),
                names=["Idx", f"Val{seed}"],
                usecols=data_cols # type: ignore
            ) for seed in range(seeds)
        ]

        merge_method = partial(pd.merge, how="outer", on="Idx")
        merged_df = reduce(merge_method, ds)
        
        seed_columns = [f"Val{seed}" for seed in range(seeds)]
        merged_df["ValMean"] = merged_df[seed_columns].mean(axis=1)
        merged_df["ValStd"] = merged_df[seed_columns].std(axis=1)

        dfs[run_name] = merged_df
    
    return dfs


def get_data_dataset(result_path: str, runs: Mapping[str, MimicAgent | GameModelAgent], observer_type: type[Observer], seeds: int = 10, processes: int = 4, epochs: int = 100, shuffle: bool = True, y: bool = False) -> dict[str, pd.DataFrame] | None:
    dfs: dict[str, pd.DataFrame] = {}

    # `None` means all seeds are missing, and so a new dataframe has to be created
    missing: dict[str, list[int] | None] = {}
    for run_name in runs:
        df_path = f"{result_path}_{run_name}.csv"
        if path.exists(df_path):
            df = pd.read_csv(df_path)
            dfs[run_name] = df
            missing_seeds = sorted({seed for seed in range(seeds) for attribute in observer_type.attributes() if f"{attribute}{seed}" not in df.columns})
            if missing_seeds:
                missing[run_name] = missing_seeds

        else:
            missing[run_name] = None

    if missing:
        if not y:
            names_list = [f"- {run_name} (seeds: {seeds})" for run_name, seeds in missing.items()]
            print("The following runs are missing:", *names_list, sep="\n")
            ans = input("Do you want to run them now? [y/N] ")
            if ans.upper() != "Y":
                return None

        # Collect the data

        with mp.get_context("spawn").Pool(processes=processes) as pool:
            dataset_run_partial = partial(dataset_run_df, epochs=epochs, shuffle=shuffle)

            args: list[tuple[str, FootsiesAgentBase, str, type[Observer], int]] = []
            for i, (run_name, missing_seeds) in enumerate(missing.items()):
                agent = runs[run_name]

                if missing_seeds is None:
                    missing_seeds = list(range(seeds))

                for seed in missing_seeds:
                    save_path = f"{result_path}_{run_name}_S{seed}_O.csv"
                    a = (save_path, agent, run_name, observer_type, seed)
                    args.append(a)

            dfs_flat: list[pd.DataFrame] = pool.starmap(dataset_run_partial, args)

        # Create/update the dataframes with the observer data

        i = 0
        for run_name, missing_seeds in missing.items():
            # If None, then we want to consider all seeds
            if missing_seeds is None:
                missing_seeds = list(range(seeds))

            # Get the dataframes pertaining to this run only
            missing_seeds_n = len(missing_seeds)
            dfs_to_merge = dfs_flat[i:i+missing_seeds_n]
            i += missing_seeds_n

            # If there is already a dataframe, then merge the new ones with it
            if run_name in dfs:
                dfs_to_merge.append(dfs[run_name])
            
            # Perform the merge...
            merge_method = partial(pd.merge, how="outer", on="Idx")
            merged_df = reduce(merge_method, dfs_to_merge)

            # ... and store
            dfs[run_name] = merged_df

        # Save the data for posterity

        for run_name in missing:
            df = dfs[run_name]
            df.to_csv(f"{result_path}_{run_name}.csv", index=False)
    
    # Calculate aggregations according to the seeds

    for _, df in dfs.items():
        for attribute in observer_type.attributes():
            attribute_seed_columns = [f"{attribute}{seed}" for seed in range(seeds)]
            df[f"{attribute}Mean"] = df[attribute_seed_columns].mean(axis=1)
            df[f"{attribute}Std"] = df[attribute_seed_columns].std(axis=1)

    return dfs
