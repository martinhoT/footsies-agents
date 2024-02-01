import argparse
from dataclasses import dataclass
import os


@dataclass
class MainArgs:
    episodes: int
    time_steps: int
    penalize_truncation: float

    misc: "MiscArgs"
    agent: "AgentArgs"
    env: "EnvArgs"
    self_play: "SelfPlayArgs"


@dataclass
class MiscArgs:
    save: bool
    load: bool
    log: bool
    log_dir: str
    log_frequency: int
    log_test_states_number: int
    log_step_start: int


@dataclass
class AgentArgs:
    kwargs: dict
    name: str
    model_name: str
    is_sb3: bool


@dataclass
class EnvArgs:
    kwargs: dict
    name: str
    is_footsies: bool
    wrapper_time_limit: int
    footsies_wrapper_norm: bool
    footsies_wrapper_acd: bool
    footsies_wrapper_fs: bool


@dataclass
class SelfPlayArgs:
    enabled: bool
    snapshot_freq: int
    max_snapshots: int
    mix_bot: int


def extract_kwargs(n_kwargs: dict, s_kwargs: dict, b_kwargs: dict) -> dict:
    kwargs = {}
    if n_kwargs is not None:
        if len(n_kwargs) % 2 != 0:
            raise ValueError(
                "the values passed to '--[...]-F-kwargs' should be a list of key-value pairs"
            )

        def convert_to_int_or_float(number):
            try:
                res = int(number)
            except ValueError:
                res = float(number)

            return res

        kwargs.update(
            {
                k: convert_to_int_or_float(v)
                for k, v in zip(n_kwargs[0::2], n_kwargs[1::2])
            }
        )

    if s_kwargs is not None:
        if len(s_kwargs) % 2 != 0:
            raise ValueError(
                "the values passed to '--[...]-S-kwargs' should be a list of key-value pairs"
            )

        kwargs.update(dict(zip(s_kwargs[0::2], s_kwargs[1::2])))

    if b_kwargs is not None:
        if len(b_kwargs) % 2 != 0:
            raise ValueError(
                "the values passed to '--[...]-B-kwargs' should be a list of key-value pairs"
            )

        for k, v in zip(b_kwargs[0::2], b_kwargs[1::2]):
            v_lower = v.lower()
            if v_lower == "true":
                kwargs[k] = True
            elif v_lower == "false":
                kwargs[k] = False
            else:
                raise ValueError(
                    f"the value passed to key '{k}' on the '--[...]-B-kwargs' kwarg list is not a boolean ('{v}' is not 'true' or 'false')"
                )

    return kwargs


def parse_args() -> MainArgs:
    available_agents = [
        file.name
        for file in os.scandir("agents")
        if file.is_dir() and file.name != "__pycache__"
    ]
    available_agents_str = ", ".join(available_agents)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "agent",
        type=str,
        help=f"agent implementation to use (available: {available_agents_str}). If name is in the form 'sb3.<agent>', then the Stable-Baselines3 algorithm <agent> will be used instead",
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        default="FOOTSIES",
        help="Gymnasium environment to use. The special value 'FOOTSIES' instantiates the FOOTSIES environment",
    )
    parser.add_argument(
        "-eN",
        "--env-N-kwargs",
        action="extend",
        nargs="+",
        type=str,
        help="key-value pairs to pass as keyword arguments to the environment. Values are treated as numbers",
    )
    parser.add_argument(
        "-eS",
        "--env-S-kwargs",
        action="extend",
        nargs="+",
        type=str,
        help="key-value pairs to pass as keyword arguments to the environment. Values are treated as strings",
    )
    parser.add_argument(
        "-eB",
        "--env-B-kwargs",
        action="extend",
        nargs="+",
        type=str,
        help="key-value pairs to pass as keyword arguments to the environment. Values are treated as booleans",
    )
    parser.add_argument(
        "--footsies-path",
        type=str,
        default=None,
        help="location of the FOOTSIES executable. Only required if using the FOOTSIES environment",
    )
    parser.add_argument(
        "--footsies-wrapper-norm",
        action="store_true",
        help="use the Normalized wrapper for FOOTSIES. Only has an effect when using the FOOTSIES environment",
    )
    parser.add_argument(
        "--footsies-wrapper-acd",
        action="store_true",
        help="use the Action Combinations Discretized wrapper for FOOTSIES. Only has an effect when using the FOOTSIES environment",
    )
    parser.add_argument(
        "--footsies-wrapper-fs",
        action="store_true",
        help="use the Frame Skipped wrapper for FOOTSIES. Only has an effect when using the FOOTSIES environment",
    )
    parser.add_argument(
        "--footsies-self-play",
        action="store_true",
        help="use self-play during training on the FOOTSIES environment. It's recommended to use the time limit wrapper",
    )
    parser.add_argument(
        "--footsies-self-play-snapshot-freq",
        type=int,
        default=1000,
        help="the frequency with which to take snapshots of the current agent for the opponent pool, in number of episodes",
    )
    parser.add_argument(
        "--footsies-self-play-max-snapshots",
        type=int,
        default=100,
        help="maximum number of snapshots to hold at once in the opponent pool",
    )
    parser.add_argument(
        "--footsies-self-play-mix-bot",
        type=int,
        default=None,
        help="the frequency, in number of episodes, with which the opponent during self-play will be the in-game bot",
    )
    parser.add_argument(
        "--footsies-self-play-port",
        type=int,
        default=11001,
        help="port to be used for the opponent socket",
    )
    parser.add_argument(
        "--wrapper-time-limit",
        type=int,
        default=99
        * 60,  # NOTE: not actually sure if it's 60, for FOOTSIES it may be 50
        help="add a time limit wrapper to the environment, with the time limit being enforced after the given number of time steps. Defaults to a number equivalent to 99 seconds in FOOTSIES",
    )
    parser.add_argument("--episodes", type=int, default=None, help="number of episodes. Will be ignored if an SB3 agent is used")
    parser.add_argument("--time-steps", type=int, default=None, help="number of time steps. Will be ignored if a FOOTSIES agent is used")
    parser.add_argument("--penalize-truncation", type=float, default=None, help="how much to penalize the agent in case the environment is truncated, useful when a time limit is defined for instance. No penalization by default")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="if passed, the model won't be saved to disk after training",
    )
    parser.add_argument(
        "--no-load",
        action="store_true",
        help="if passed, the model won't be loaded from disk before training",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="the name of the model for saving and loading",
        default=None,
    )
    parser.add_argument(
        "-mN",
        "--model-N-kwargs",
        action="extend",
        nargs="+",
        type=str,
        help="key-value pairs to pass as keyword arguments to the agent implementation. Values are treated as numbers",
    )
    parser.add_argument(
        "-mS",
        "--model-S-kwargs",
        action="extend",
        nargs="+",
        type=str,
        help="key-value pairs to pass as keyword arguments to the agent implementation. Values are treated as strings",
    )
    parser.add_argument(
        "-mB",
        "--model-B-kwargs",
        action="extend",
        nargs="+",
        type=str,
        help="key-value pairs to pass as keyword arguments to the agent implementation. Values are treated as booleans",
    )
    parser.add_argument(
        "--no-log", action="store_true", help="if passed, the model won't be logged"
    )
    parser.add_argument(
        "--log-frequency",
        type=int,
        default=5000,
        help="number of time steps between each log",
    )
    parser.add_argument(
        "--log-test-states-number",
        type=int,
        default=5000,
        help="number of test states to use when evaluating some metrics for logging",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="directory to which Tensorboard logs will be written",
    )
    parser.add_argument(
        "--log-step-start",
        type=int,
        default=0,
        help="value at which the logging time step will start, useful for appending to existing logs",
    )

    args = parser.parse_args()

    # Prepare various variables, including keyword arguments
    env_kwargs = extract_kwargs(args.env_N_kwargs, args.env_S_kwargs, args.env_B_kwargs)
    model_kwargs = extract_kwargs(
        args.model_N_kwargs, args.model_S_kwargs, args.model_B_kwargs
    )

    is_sb3 = args.agent.startswith("sb3.")
    is_footsies = args.env == "FOOTSIES"
    will_footsies_self_play = args.footsies_self_play and is_footsies

    if is_sb3:
        if args.episodes is not None:
            print("WARN: specifying a number of episodes for SB3 algorithms is not supported, will be ignored")

    if will_footsies_self_play:
        # Set dummy opponent for now, and set later with a copy of the instanced agent
        env_kwargs["opponent"] = lambda o: (False, False, False)
        env_kwargs["opponent_port"] = args.footsies_self_play_port

    if is_footsies and args.footsies_path is None:
        raise ValueError(
            "the path to the FOOTSIES executable should be specified with '--footsies-path' when using the FOOTSIES environment"
        )

    env_kwargs["game_path"] = args.footsies_path

    return MainArgs(
        episodes=args.episodes,
        time_steps=args.time_steps,
        penalize_truncation=args.penalize_truncation,
        misc=MiscArgs(
            save=not args.no_save,
            load=not args.no_load,
            log=not args.no_log,
            log_dir=args.log_dir,
            log_frequency=args.log_frequency,
            log_test_states_number=args.log_test_states_number,
            log_step_start=args.log_step_start,
        ),
        agent=AgentArgs(
            kwargs=model_kwargs,
            name=args.agent[4:] if is_sb3 else args.agent,
            model_name=args.agent if args.model_name is None else args.model_name,
            is_sb3=is_sb3,
        ),
        env=EnvArgs(
            kwargs=env_kwargs,
            name=args.env,
            is_footsies=is_footsies,
            wrapper_time_limit=args.wrapper_time_limit,
            footsies_wrapper_norm=args.footsies_wrapper_norm,
            footsies_wrapper_acd=args.footsies_wrapper_acd,
            footsies_wrapper_fs=args.footsies_wrapper_fs,
        ),
        self_play=SelfPlayArgs(
            enabled=will_footsies_self_play,
            snapshot_freq=args.footsies_self_play_snapshot_freq,
            max_snapshots=args.footsies_self_play_max_snapshots,
            mix_bot=args.footsies_self_play_mix_bot,
        )
    )