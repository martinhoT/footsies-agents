import pandas as pd
import multiprocessing as mp
from typing import Sequence, Mapping
from agents.base import FootsiesAgentBase
from footsies_gym.envs.footsies import FootsiesEnv
from agents.game_model.agent import GameModelAgent
from agents.mimic.agent import MimicAgent
from args import MainArgs
from os import path
from scripts.evaluation.custom_loop import Observer, custom_loop, dataset_run, AgentCustomRun
from functools import partial
from main import main
from dataclasses import replace
import logging


LOGGER = logging.getLogger("scripts.evaluation.data_collectors")


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

        with mp.Pool(processes=processes) as pool:
            custom_loop_partial = partial(custom_loop, timesteps=timesteps)

            args: list[tuple[AgentCustomRun, str, int, type[Observer], int]] = []
            for i, (run_name, missing_seeds) in enumerate(missing.items()):
                run_args = runs[run_name]

                if missing_seeds is None:
                    missing_seeds = list(range(seeds))

                for seed in missing_seeds:
                    a = (run_args, run_name, i * seeds + seed, observer_type, seed)
                    args.append(a)            

            observers_flat: list[Observer] = pool.starmap(custom_loop_partial, args)
        
        # Batch the observers according to the run they belong to

        observers: list[tuple[Observer, ...]] = []
        i = 0
        for run_name, missing_seeds in missing.items():
            missing_seeds_n = len(missing_seeds) if missing_seeds is not None else seeds
            run_observers = tuple(o for o in observers_flat[i:i+missing_seeds_n] if o is not None)
            observers.append(run_observers)
            i += missing_seeds_n

        # Create dataframes with the data

        for run_observers, (run_name, missing_seeds) in zip(observers, missing.items()):
            # Create the dataframe from scratch
            if missing_seeds is None:
                # The indices should be the same among all seeds, so we can just get from the first observer
                indices, _ = run_observers[0].data

                data: dict[str, Sequence[int | float]] = {"Idx": indices}
                for observer, seed in zip(run_observers, range(seeds)):
                    _, values = observer.data
                    for attribute, value in zip(observer.attributes(), values):
                        data[f"{attribute}{seed}"] = value

                df = pd.DataFrame(data)
                dfs[run_name] = df

            # Otherwise, populate existing dataframe
            else:
                df = dfs[run_name]
                for observer, seed in zip(run_observers, missing_seeds):
                    _, values = observer.data
                    for attribute, value in zip(observer.attributes(), values):
                        df[f"{attribute}{seed}"] = value

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


def get_data(data: str, runs: dict[str, MainArgs], seeds: int = 10, processes: int = 4, y: bool = False, data_cols: Sequence[int] = (0, 1)) -> dict[str, pd.DataFrame] | None:
    # Halve the number of processes since there are technically going to be two processes per run: the agent and the game
    processes //= 2
    
    missing: dict[str, list[tuple[str, int]]] = {}
    for run_name in runs:
        for seed in range(seeds):
            run_fullname = f"eval_{run_name}_S{seed}"
            data_path = path.join("runs", run_fullname)
            if not path.exists(data_path):
                missing.setdefault(run_name, []).append((run_fullname, seed))

    if missing:
        if not y:
            names_list = [f"- {run_name} (seeds: {[seed for _, seed in missing_seeded_runs]})" for run_name, missing_seeded_runs in missing.items()]
            print("The following runs are missing:", *names_list, sep="\n")
            ans = input("Do you want to run them now? [y/N] ")
            if ans.upper() != "Y":
                return None
        
        # Try to avoid oversubscription, since each agent will have at least 2 processes running: itself, and the game
        with mp.Pool(processes=processes) as pool:
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

                    # Update main args with everything + seed
                    run_args_modified = replace(run_args,
                        env=env_args_modified,
                        agent=agent_args_modified,
                        seed=seed,
                    )
                    
                    args.append(run_args_modified)

            pool.map(main, args)
    
    dfs: dict[str, pd.DataFrame] = {}
    for run_name in runs:
        df = pd.DataFrame([], columns=["Idx", "ValMean", "ValStd"])
        seed_columns = [f"Val{seed}" for seed in range(seeds)]

        for seed in range(seeds):
            data_path = path.join("runs", f"eval_{run_name}_S{seed}", f"{data}.csv")
            d = pd.read_csv(
                data_path,
                names=["Idx", "Val"],
                usecols=data_cols # type: ignore
            )
            df["Idx"] = d["Idx"]
            df[f"Val{seed}"] = d["Val"]
        
        df["ValMean"] = df[seed_columns].mean(axis=1)
        df["ValStd"] = df[seed_columns].std(axis=1)

        dfs[run_name] = df
    
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

        with mp.Pool(processes=processes) as pool:
            dataset_run_partial = partial(dataset_run, epochs=epochs, shuffle=shuffle)

            args: list[tuple[FootsiesAgentBase, str, type[Observer], int]] = []
            for i, (run_name, missing_seeds) in enumerate(missing.items()):
                agent = runs[run_name]

                if missing_seeds is None:
                    missing_seeds = list(range(seeds))

                for seed in missing_seeds:
                    a = (agent, run_name, observer_type, seed)
                    args.append(a)

            observers_flat: list[Observer] = pool.starmap(dataset_run_partial, args)

        # Batch the observers according to the run they belong to

        observers: list[tuple[Observer, ...]] = []
        i = 0
        for run_name, missing_seeds in missing.items():
            missing_seeds_n = len(missing_seeds) if missing_seeds is not None else seeds
            run_observers = tuple(o for o in observers_flat[i:i+missing_seeds_n] if o is not None)
            observers.append(run_observers)
            i += missing_seeds_n

        # Create dataframes with the data

        for run_observers, (run_name, missing_seeds) in zip(observers, missing.items()):
            # Create the dataframe from scratch
            if missing_seeds is None:
                # The indices should be the same among all seeds, so we can just get from the first observer
                indices, _ = run_observers[0].data

                data: dict[str, Sequence[int | float]] = {"Idx": indices}
                for observer, seed in zip(run_observers, range(seeds)):
                    _, values = observer.data
                    for attribute, value in zip(observer.attributes(), values):
                        data[f"{attribute}{seed}"] = value

                df = pd.DataFrame(data)
                dfs[run_name] = df

            # Otherwise, populate existing dataframe
            else:
                df = dfs[run_name]
                for observer, seed in zip(run_observers, missing_seeds):
                    _, values = observer.data
                    for attribute, value in zip(observer.attributes(), values):
                        df[f"{attribute}{seed}"] = value

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
