import argparse
import os
import logging
from dataclasses import dataclass
from agents.torch_utils import hidden_layer_parameters_from_specifications
from intrinsic.base import IntrinsicRewardScheme
from intrinsic.counts import CountBasedScheme
from intrinsic.icm import ICMScheme
from intrinsic.rnd import RNDScheme


@dataclass
class MainArgs:
    episodes: int
    time_steps: int
    penalize_truncation: float
    curriculum: bool
    intrinsic_reward_scheme: type[IntrinsicRewardScheme] | None

    misc: "MiscArgs"
    agent: "AgentArgs"
    env: "EnvArgs"
    self_play: "SelfPlayArgs"


@dataclass
class MiscArgs:
    save: bool
    load: bool
    log: bool
    log_frequency: int
    log_test_states_number: int
    log_step_start: int
    log_episode_start: int
    log_file_level: int
    hogwild: bool
    hogwild_cpus: int
    hogwild_n_workers: int


@dataclass
class AgentArgs:
    kwargs: dict
    model: str
    name: str
    is_sb3: bool


@dataclass
class EnvArgs:
    kwargs: dict
    name: str
    is_footsies: bool
    wrapper_time_limit: int
    footsies_wrapper_norm: bool
    footsies_wrapper_norm_guard: bool
    footsies_wrapper_acd: bool
    footsies_wrapper_fs: bool
    footsies_wrapper_adv: bool
    footsies_wrapper_phasic: bool
    diayn: bool
    diayn_kwargs: dict
    torch: bool


@dataclass
class SelfPlayArgs:
    enabled: bool
    max_opponents: int
    snapshot_interval: int
    switch_interval: int
    mix_bot: int
    add_curriculum_opps: bool


@dataclass
class ExperimentArgs:
    agent_name: str
    total_episodes: int
    study_name: str
    direction: str
    n_trials: int


def extract_kwargs(n_kwargs: list[str], s_kwargs: list[str], b_kwargs: list[str]) -> dict:
    kwargs = {}
    if n_kwargs is not None:
        if len(n_kwargs) % 2 != 0:
            raise ValueError(
                "the values passed to '--[...]-N-kwargs' should be a list of key-value pairs"
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


def add_agent_argument(parser: argparse.ArgumentParser):
    available_agents = [
        file.name[:-3] # remove the .py extension
        for file in os.scandir("models")
        if file.is_dir() and file.name != "__pycache__"
    ]
    available_agents_str = ", ".join(available_agents)
    
    parser.add_argument(
        "agent",
        type=str,
        help=f"agent initializer/model to use (available: {available_agents_str}). If name is in the form 'sb3.<agent>', then the Stable-Baselines3 algorithm <agent> will be used instead",
    )


def parse_args() -> MainArgs:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_agent_argument(parser)
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
        "--footsies-wrapper-norm-no-guard",
        action="store_true",
        help="for the Normalized wrapper, disable normalization of the guard variable. Only has an effect if using the Normalized wrapper",
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
        "--footsies-wrapper-adv",
        action="store_true",
        help="use the Encourage Advance wrapper for FOOTSIES. Only has an effect when using the FOOTSIES environment",
    )
    parser.add_argument(
        "--footsies-wrapper-phasic",
        action="store_true",
        help="use the Phasic Move Progress wrapper for FOOTSIES. Only has an effect when using the FOOTSIES environment",
    )
    parser.add_argument(
        "--footsies-curriculum",
        action="store_true",
        help="perform curriculum learning with pre-made rule-based opponents"
    )
    parser.add_argument(
        "--footsies-self-play",
        action="store_true",
        help="use self-play during training on the FOOTSIES environment. It's recommended to use the time limit wrapper",
    )
    parser.add_argument(
        "--footsies-self-play-add-curriculum-opps",
        action="store_true",
        help="add all opponents that are used for curriculum learning into the self-play's opponent pool"
    )
    parser.add_argument(
        "--footsies-self-play-max-opponents",
        type=int,
        default=10,
        help="maximum number of opponents to hold at once in the opponent pool. Doesn't count the in-game bot",
    )
    parser.add_argument(
        "--footsies-self-play-snapshot-interval",
        type=int,
        default=2000,
        help="the interval between snapshots of the current policy for the opponent pool, in number of episodes",
    )
    parser.add_argument(
        "--footsies-self-play-switch-interval",
        type=int,
        default=100,
        help="the interval between opponent switched, in number of episodes"
    )
    parser.add_argument(
        "--footsies-self-play-mix-bot",
        type=int,
        default=1,
        help="how many opponents will the in-game opponent count as, when sampling from the opponent pool. In-game bot won't be sampled if 0",
    )
    parser.add_argument(
        "--wrapper-time-limit",
        type=int,
        default=99 * 60,  # NOTE: not actually sure if it's 60, for FOOTSIES it may be 50
        help="add a time limit wrapper to the environment, with the time limit being enforced after the given number of time steps. Defaults to a number equivalent to 99 seconds in FOOTSIES",
    )
    parser.add_argument(
        "--intrinsic-reward",
        choices=["count", "icm", "rnd", "none"],
        default="none",
        help="the type of intrinsic reward to use. 'none' means no intrinsic reward will be used"
    )
    parser.add_argument(
        "--diayn",
        action="store_true",
        help="use the DIAYN wrapper, which replaces the environment's reward with the pseudo-reward from DIAYN, fostering the creation of a diverse set of task-agnostic skills"
    )
    parser.add_argument(
        "--diayn-skill-dim",
        type=int,
        default=5,
        help="the dimensionality of the skill vectors for DIAYN, roughly equivalent to the desired number of skills to learn"
    )
    parser.add_argument(
        "--diayn-no-baseline",
        action="store_true",
        help="whether to exclude the baseline in the pseudo-reward of DIAYN. Excluding it encourages haste rather than staying alive"
    )
    parser.add_argument(
        "--diayn-discriminator-learning-rate",
        type=float,
        default=1e-3,
        help="the learning rate for the discriminator network in DIAYN"
    )
    parser.add_argument(
        "--diayn-discriminator-hidden-layer-sizes-specification",
        type=str,
        default="32",
        help="specification of the hidden layer sizes for the discriminator network in DIAYN. Should be a string of comma-separated integers"
    )
    parser.add_argument(
        "--diayn-discriminator-hidden-layer-activation-specification",
        type=str,
        default="ReLU",
        help="specification of the hidden layer activation for the discriminator network in DIAYN. Should be a string of the name of a PyTorch activation function"
    )
    parser.add_argument(
        "--torch",
        action="store_true",
        help="whether to transform environment observations to torch tensors"
    )
    parser.add_argument("--episodes", type=int, default=None, help="number of episodes. Will be ignored if an SB3 agent is used")
    parser.add_argument("--time-steps", type=int, default=None, help="number of time steps. Will be ignored if a FOOTSIES agent is used")
    parser.add_argument("--penalize-truncation", type=float, default=None, help="how much to penalize the agent in case the environment is truncated, useful when a time limit is defined for instance. No penalization by default")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="if passed, the agent won't be saved to disk after training",
    )
    parser.add_argument(
        "--no-load",
        action="store_true",
        help="if passed, the agent won't be loaded from disk before training",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="the name of the agent, for saving, loading and logging",
        default=None,
    )
    parser.add_argument(
        "-mN",
        "--agent-N-kwargs",
        action="extend",
        nargs="+",
        type=str,
        help="key-value pairs to pass as keyword arguments to the agent initializer. Values are treated as numbers",
    )
    parser.add_argument(
        "-mS",
        "--agent-S-kwargs",
        action="extend",
        nargs="+",
        type=str,
        help="key-value pairs to pass as keyword arguments to the agent initializer. Values are treated as strings",
    )
    parser.add_argument(
        "-mB",
        "--agent-B-kwargs",
        action="extend",
        nargs="+",
        type=str,
        help="key-value pairs to pass as keyword arguments to the agent initializer. Values are treated as booleans",
    )
    parser.add_argument(
        "--no-log", action="store_true", help="if passed, the agent won't be logged (neither standard not Tensorboard logs)"
    )
    parser.add_argument(
        "--log-frequency",
        type=int,
        default=5000,
        help="number of time steps between each Tensorboard log",
    )
    parser.add_argument(
        "--log-test-states-number",
        type=int,
        default=5000,
        help="number of test states to use when evaluating some metrics for Tensorboard logging",
    )
    parser.add_argument(
        "--log-step-start",
        type=int,
        default=0,
        help="value at which the logging time step will start, useful for appending to existing Tensorboard logs",
    )
    parser.add_argument(
        "--log-episode-start",
        type=int,
        default=0,
        help="value at which the logging episode will start, useful for appending to existing Tensorboard logs",
    )
    parser.add_argument(
        "--log-file-level",
        choices=["critical", "error", "warning", "info", "debug"],
        default="debug",
        help="the log level of the logs created in the log file. Recommended to be either debug or info"
    )
    parser.add_argument(
        "--hogwild",
        action="store_true",
        help="whether to use the Hogwild! asynchronous training algorithm. Only available for FOOTSIES agents based on PyTorch (for sharing of model parameters)"
    )
    parser.add_argument(
        "--hogwild-cpus",
        type=int,
        default=None,
        help="maximum number of CPUs to use for Hogwild! training. If None, will use all available",
    )
    parser.add_argument(
        "--hogwild-n-workers",
        type=int,
        default=6,
        help="number of parallel workers to use for Hogwild! training",
    )

    args = parser.parse_args()

    # Prepare various variables, including keyword arguments
    env_kwargs = extract_kwargs(args.env_N_kwargs, args.env_S_kwargs, args.env_B_kwargs)
    model_kwargs = extract_kwargs(
        args.agent_N_kwargs, args.agent_S_kwargs, args.agent_B_kwargs
    )

    is_sb3 = args.agent.startswith("sb3.")
    is_footsies = args.env == "FOOTSIES"
    will_footsies_self_play = args.footsies_self_play and is_footsies
    will_footsies_curriculum = args.footsies_curriculum and is_footsies

    if is_sb3:
        if args.episodes is not None:
            print("WARNING: specifying a number of episodes for SB3 algorithms is not supported, will be ignored")

    if will_footsies_self_play or will_footsies_curriculum:
        # Set dummy opponent for now, and set later with a custom opponent
        env_kwargs["opponent"] = lambda o: (False, False, False)

    elif will_footsies_self_play and will_footsies_curriculum:
        raise ValueError("can't use both self-play and curriculum learning at the same time")

    if is_footsies:
        if args.footsies_path is None:
            raise ValueError(
                "the path to the FOOTSIES executable should be specified with '--footsies-path' when using the FOOTSIES environment"
            )
        
        env_kwargs["game_path"] = args.footsies_path

    diayn_discriminator_hidden_layer_sizes, diayn_discriminator_hidden_layer_activation = hidden_layer_parameters_from_specifications(
        args.diayn_discriminator_hidden_layer_sizes_specification,
        args.diayn_discriminator_hidden_layer_activation_specification,
    )

    
    intrinsic_reward_scheme = {
        "count": CountBasedScheme,
        "icm": ICMScheme,
        "rnd": RNDScheme,
        "none": None,
    }[args.intrinsic_reward]

    return MainArgs(
        episodes=args.episodes,
        time_steps=args.time_steps,
        penalize_truncation=args.penalize_truncation,
        curriculum=args.footsies_curriculum,
        intrinsic_reward_scheme=intrinsic_reward_scheme,
        misc=MiscArgs(
            save=not args.no_save,
            load=not args.no_load,
            log=not args.no_log,
            log_frequency=args.log_frequency,
            log_test_states_number=args.log_test_states_number,
            log_step_start=args.log_step_start,
            log_episode_start=args.log_episode_start,
            log_file_level=getattr(logging, args.log_file_level.upper()),
            hogwild=args.hogwild,
            hogwild_cpus=args.hogwild_cpus,
            hogwild_n_workers=args.hogwild_n_workers,
        ),
        agent=AgentArgs(
            kwargs=model_kwargs,
            model=args.agent[4:] if is_sb3 else args.agent,
            name=args.name if args.name is not None else args.agent,
            is_sb3=is_sb3,
        ),
        env=EnvArgs(
            kwargs=env_kwargs,
            name=args.env,
            is_footsies=is_footsies,
            wrapper_time_limit=args.wrapper_time_limit,
            footsies_wrapper_norm=args.footsies_wrapper_norm,
            footsies_wrapper_norm_guard=not args.footsies_wrapper_norm_no_guard,
            footsies_wrapper_acd=args.footsies_wrapper_acd,
            footsies_wrapper_fs=args.footsies_wrapper_fs,
            footsies_wrapper_adv=args.footsies_wrapper_adv,
            footsies_wrapper_phasic=args.footsies_wrapper_phasic,
            diayn=args.diayn,
            diayn_kwargs={
                "skill_dim": args.diayn_skill_dim,
                "include_baseline": not args.diayn_no_baseline,
                "discriminator_learning_rate": args.diayn_discriminator_learning_rate,
                "discriminator_hidden_layer_sizes": diayn_discriminator_hidden_layer_sizes,
                "discriminator_hidden_layer_activation": diayn_discriminator_hidden_layer_activation,
                "log_frequency": args.log_frequency,
            },
            torch=args.torch,
        ),
        self_play=SelfPlayArgs(
            enabled=will_footsies_self_play,
            max_opponents=args.footsies_self_play_max_opponents,
            snapshot_interval=args.footsies_self_play_snapshot_interval,
            switch_interval=args.footsies_self_play_switch_interval,
            mix_bot=args.footsies_self_play_mix_bot,
            add_curriculum_opps=args.footsies_self_play_add_curriculum_opps,
        )
    )


def parse_args_experiment() -> ExperimentArgs:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_agent_argument(parser)
    parser.add_argument("-e", "--episodes", type=int, default=15000, help="number of episodes")
    parser.add_argument("-s", "--study-name", type=str, default="test-experiment", help="name of the study (should be the same among processes on the same study)")
    parser.add_argument("--maximize", action="store_true", help="maximize the objective value. If not specified, will minimize")
    parser.add_argument("-n", "--n-trials", type=int, default=10, help="the number of trials to attempt")

    args = parser.parse_args()

    return ExperimentArgs(
        agent_name=args.agent,
        total_episodes=args.episodes,
        study_name=args.study_name,
        direction="maximize" if args.maximize else "minimize",
        n_trials=args.n_trials,
    )
