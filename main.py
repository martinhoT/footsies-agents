import os
import importlib
import gymnasium as gym
import numpy as np
import torch as T
import logging
import json
import datetime
import random
from gymnasium import Env
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from gymnasium.wrappers.time_limit import TimeLimit
from agents.wrappers import FootsiesSimpleActions, FootsiesEncourageAdvance, FootsiesPhasicMoveProgress, AppendSimpleHistoryWrapper
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.envs.exceptions import FootsiesGameClosedError
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.frame_skip import FootsiesFrameSkipped
from tqdm import tqdm
from itertools import count
from copy import deepcopy
from functools import partial
from typing import Callable, Iterable, cast
from stable_baselines3.common.base_class import BaseAlgorithm
from agents.base import FootsiesAgentBase, FootsiesAgentTorch
from agents.diayn import DIAYN, DIAYNWrapper
from agents.logger import TrainingLoggerWrapper
from agents.torch_utils import hogwild
from args import MainArgs, parse_args, EnvArgs
from opponents.self_play import SelfPlayManager
from opponents.curriculum import BSpecialSpammer, CurriculumManager, BSpammer, NSpecialSpammer, NSpammer, WhiffPunisher
from opponents.base import OpponentManager
from intrinsic.base import IntrinsicRewardScheme
from agents.action import ActionMap
from dataclasses import asdict
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
from gymnasium.spaces import Discrete
from models import ModelInit
from torch.utils.tensorboard import SummaryWriter # type: ignore
from agents.wrappers import OpponentManagerWrapper

LOGGER = logging.getLogger("main")


def import_sb3(model: str, env: Env, parameters: dict) -> BaseAlgorithm:
    import stable_baselines3
    agent_class = stable_baselines3.__dict__[model.upper()]
    return agent_class(
        policy="MlpPolicy", # we assume we will never need the CnnPolicy
        env=env,
        **parameters,
    )


def import_agent(model: str, env: Env, parameters: dict) -> tuple[FootsiesAgentBase, dict[str, list]]:
    models = importlib.import_module("models")
    model_init: ModelInit = getattr(models, model + "_")

    assert env.observation_space.shape
    assert isinstance(env.action_space, Discrete)

    agent, loggables = model_init(
        observation_space_size=env.observation_space.shape[0],
        action_space_size=int(env.action_space.n),
        **parameters
    )

    return agent, loggables


def load_agent(agent: FootsiesAgentBase | BaseAlgorithm, name: str, folder: str = "saved") -> bool:
    """Load the trained parameters of the `agent` from disk."""
    agent_path = os.path.join(folder, name)
    is_footsies_agent = isinstance(agent, FootsiesAgentBase)
    if not is_footsies_agent:
        agent_path = os.path.join(agent_path, name + ".zip")

    if os.path.exists(agent_path):
        if is_footsies_agent and not os.path.isdir(agent_path):
            raise OSError(f"the existing file '{agent_path}' is not a folder!")

        if is_footsies_agent:
            agent.load(agent_path)
        else:
            agent.set_parameters(agent_path)

        LOGGER.info("Agent '%s' loaded", name)
        return True

    LOGGER.info("Can't load agent '%s', there was no agent saved!", name)    
    return False


def save_agent(agent: FootsiesAgentBase | BaseAlgorithm, name: str, folder: str = "saved"):
    """Save the trained parameters of the `agent` to disk."""
    agent_folder_path = os.path.join(folder, name)
    is_footsies_agent = isinstance(agent, FootsiesAgentBase)

    if not os.path.exists(agent_folder_path):
        os.makedirs(agent_folder_path)

    # Both FOOTSIES and SB3 agents use the same method and signature (mostly).
    # FOOTSIES agents expect a folder, and SB3 agents expect the actual file.
    if is_footsies_agent:
        agent.save(agent_folder_path)
    else:
        agent_path = os.path.join(agent_folder_path, name)
        agent.save(agent_path)
    LOGGER.info("Agent '%s' saved", name)


def load_agent_parameters(name: str, folder: str = "saved") -> dict:
    """Load the agent initialization parameters from disk."""
    agent_folder_path = os.path.join(folder, name)
    
    # Save the parameters used to instantiate this agent
    with open(os.path.join(agent_folder_path, "parameters.json"), "rt") as f:
        return json.load(f)


def save_agent_parameters(parameters: dict, name: str, folder: str = "saved"):
    """Save the agent initialization parameters to disk."""
    agent_folder_path = os.path.join(folder, name)
    
    # Save the parameters used to instantiate this agent
    with open(os.path.join(agent_folder_path, "parameters.json"), "wt") as f:
        json.dump(parameters, f, indent=4)


def save_agent_training_args(training_args: dict, name: str, folder: str = "saved"):
    """Save the arguments passed for training the agent to disk. If there are training args already saved, then older ones will be renamed (similar to how logging does it)."""
    agent_folder_path = os.path.join(folder, name)

    timestamp = datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    file_name = os.path.join(agent_folder_path, f"training_args_{timestamp}.json")

    # Save the parameters used to instantiate this agent
    with open(file_name, "wt") as f:
        json.dump(training_args, f, indent=4)


def extract_opponent_manager(env: Env) -> OpponentManager:
    """Iterate over the environment wrappers, find the OpponentManagerWrapper and get its opponent manager. This is required, for instance, when we want to programatically know when the oppponent manager has exhausted"""
    e = env
    while e != e.unwrapped:
        if isinstance(e, OpponentManagerWrapper):
            return e.opponent_manager
        e = e.env # type: ignore

    raise ValueError("the provided wrapped environment doesn't have an opponent manager wrapper")


def train(
    agent: FootsiesAgentBase,
    env: Env,
    n_episodes: int | None = None,
    n_timesteps: int | None = None,
    penalize_truncation: float | None = None,
    intrinsic_reward_scheme: IntrinsicRewardScheme | None = None,
    episode_finished_callback: Callable[[int], None] = lambda episode: None,
    progress_bar: bool = True,
    progress_bar_kwargs: dict | None = None,
    skip_freeze: bool = True,
    initial_seed: int | None = None,
):
    """
    Train an `agent` on the given Gymnasium environment `env`.

    Parameters
    ----------
    - `agent`: implementation of FootsiesAgentBase, representing the agent
    - `env`: the Gymnasium environment to train on
    - `n_episodes`: if specified, the number of training episodes. Should not be specified along with `n_timesteps`
    - `n_timesteps`: if specified, the number of timesteps to train on. Should not be specified along with `n_episodes`
    - `penalize_truncation`: penalize the agent if the time limit was exceeded, to discourage lengthening the episode
    - `intrinsic_reward`: the intrinsic reward scheme to use, if any
    - `episode_finished_callback`: function that will be called after each episode is finished
    - `progress_bar`: whether to display a progress bar (with `tqdm`)
    - `progress_bar_kwargs`: keyword arguments to pass to the `tqdm` progress bar
    - `skip_freeze`: whether to skip environment freezes, such as histop in FOOTSIES
    - `initial_seed`: the environment seed that will be passed to the first `reset()` call
    """
    try:
        opponent_manager = extract_opponent_manager(env)
    except ValueError:
        LOGGER.info("Not using an opponent manager")
        opponent_manager = None
    else:
        LOGGER.info("Using an opponent manager")

    if n_episodes is not None and n_timesteps is not None:
        raise ValueError(f"either 'n_episodes' ({n_episodes}) or 'n_timesteps' ({n_timesteps}) is allowed to be specified, not both")

    seed = initial_seed

    agent.preprocess(env)
    LOGGER.info("Preprocessing done!")

    base_training_iterator = count() if n_episodes is None else range(n_episodes)

    training_iterator: Iterable[int]
    if progress_bar:
        total = n_episodes if n_episodes is not None else n_timesteps // 140 if n_timesteps is not None else None
        training_iterator = tqdm(base_training_iterator,
            total=total,
            unit="ep",
            colour="#80e4ed",
            dynamic_ncols=True,
            **progress_bar_kwargs, # type: ignore
        )

    else:
        training_iterator = base_training_iterator

    timestep_counter = 0

    try:
        for episode in training_iterator:
            obs, info = env.reset(seed=seed)
            
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = agent.act(obs, info)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                reward = float(reward)
                
                # Discard histop/freeze
                if skip_freeze:
                    in_hitstop = ActionMap.is_in_hitstop_ori(next_info, True) or ActionMap.is_in_hitstop_ori(next_info, False)
                    while in_hitstop and obs.isclose(next_obs).all():
                        # The agent keeps acting. For the case of the_one, this means that they have more time to react when hitstop occurs which makes sense.
                        # We are only doing this freeze-skipping thing to avoid updating the agent on meaningless transitions.
                        action = agent.act(obs, info)
                        next_obs, r, terminated, truncated, next_info = env.step(action)
                        r = float(r)
                        reward += r
                        LOGGER.debug("Skipped one transition, which is presumed to be artificial freeze")

                if penalize_truncation is not None and truncated:
                    reward = penalize_truncation
                
                if intrinsic_reward_scheme is not None:
                    # BUG: probably not info, maybe it's next_info
                    intrinsic_reward = intrinsic_reward_scheme.update_and_reward(obs, next_obs, reward, terminated, truncated, info)
                    # It's not great to use the `info` dict as the storage for intrinsic reward, but this allows the addition of such without breaking the current API.
                    # I could change it but I won't bother. Whathever agent wants to use intrinsic reward can just check if the key is present.
                    if "intrinsic_reward" in info:
                        LOGGER.warning("'intrinsic reward' key already present in info, will overwrite it although it shouldn't be present in the first place")
                    info["intrinsic_reward"] = intrinsic_reward

                agent.update(obs, next_obs, reward, terminated, truncated, info, next_info)
                obs = next_obs
                info = next_info

                timestep_counter += 1

            LOGGER.debug("Episode finished with reward %s and info %s, with termination (%s) or truncation (%s)", reward, info, terminated, truncated)

            if opponent_manager is not None and opponent_manager.exhausted:
                LOGGER.info("Opponent pool exhausted, quitting training")
                break

            episode_finished_callback(episode)

            if n_timesteps is not None and timestep_counter > n_timesteps:
                break
        
            # Only use the seed for the first episode/reset call
            seed = None

    except KeyboardInterrupt:
        LOGGER.info("Training manually interrupted (KeyboardInterrupt)")

    except FootsiesGameClosedError as e:
        LOGGER.warning("Quitting training since game closed: '%s'", e)

    except Exception as e:
        LOGGER.exception("Training stopped due to %s: '%s', ignoring and quitting training", type(e).__name__, e)
    
    finally:
        if opponent_manager is not None:
            opponent_manager.close()


def dummy_opponent(o: dict, i: dict) -> tuple[bool, bool, bool]:
    return (False, False, False)


def observation_to_torch(obs: np.ndarray) -> T.Tensor:
    return T.from_numpy(obs).float().unsqueeze(0)


def create_env(args: EnvArgs, log_dir: str | None = "runs", use_custom_opponent: bool = False) -> Env:
    # Create environment with initial wrappers
    if args.is_footsies:
        opponent = dummy_opponent if use_custom_opponent or args.self_play.enabled or args.curriculum.enabled else None

        env = FootsiesEnv(
            opponent=opponent,
            fast_forward_speed=12.0, # manually adjusted to my CPU
            **args.kwargs, # type: ignore
        )

        if args.footsies_wrapper_norm:
            if not args.footsies_wrapper_norm_guard:
                raise NotImplementedError("non-normalized guard observation variable is not supported until ActionMap (and potentially other regions of code) are slice-independent when evaluating observation regions")
            env = FootsiesNormalized(env, normalize_guard=args.footsies_wrapper_norm_guard)

        if args.footsies_wrapper_simple.enabled:
            env = FootsiesSimpleActions(env,
                agent_allow_special_moves=args.footsies_wrapper_simple.allow_agent_special_moves,
                assumed_agent_action_on_nonactionable=args.footsies_wrapper_simple.assumed_agent_action_on_nonactionable,
                assumed_opponent_action_on_nonactionable=args.footsies_wrapper_simple.assumed_opponent_action_on_nonactionable,
            )

        if args.footsies_wrapper_history:
            env = AppendSimpleHistoryWrapper(env,
                p1=cast(bool, args.footsies_wrapper_history.get("p1", True)),
                n=cast(int, args.footsies_wrapper_history.get("p1_n", 5)),
                distinct=cast(bool, args.footsies_wrapper_history.get("p1_distinct", True)),
            )
            env = AppendSimpleHistoryWrapper(env,
                p1=cast(bool, args.footsies_wrapper_history.get("p2", True)),
                n=cast(int, args.footsies_wrapper_history.get("p2_n", 5)),
                distinct=cast(bool, args.footsies_wrapper_history.get("p2_distinct", True)),
            )

        if args.footsies_wrapper_adv:
            env = FootsiesEncourageAdvance(
                env,
                log_dir=log_dir,
            )

        if args.footsies_wrapper_phasic:
            env = FootsiesPhasicMoveProgress(env)

        if args.footsies_wrapper_fs:
            raise NotImplementedError("don't use the environment's frame skipping wrapper as it's deprecated")
            env = FootsiesFrameSkipped(env)

    else:
        env = gym.make(
            args.name,
            **args.kwargs # type: ignore
        )

    # Wrap with additional, environment-independent wrappers
    if args.wrapper_time_limit > 0:
        env = TimeLimit(env, max_episode_steps=args.wrapper_time_limit)

    env = FlattenObservation(env)

    # Final FOOTSIES wrappers
    if args.is_footsies:
        if args.footsies_wrapper_acd:
            env = FootsiesActionCombinationsDiscretized(env)

    if args.torch:
        env = TransformObservation(env, observation_to_torch)

    # Final miscellaneous wrappers
    if args.diayn.enabled:
        assert env.observation_space.shape

        env = DIAYNWrapper(
            env,
            DIAYN(
                observation_dim=env.observation_space.shape[0],
                skill_dim=args.diayn.skill_dim,
                include_baseline=args.diayn.include_baseline,
                discriminator_learning_rate=args.diayn.discriminator_learning_rate,
                discriminator_hidden_layer_sizes=args.diayn.discriminator_hidden_layer_sizes,
                discriminator_hidden_layer_activation=args.diayn.discriminator_hidden_layer_activation,
            ),
            log_dir=log_dir,
        )

    # These should ideally be applied at the end only (mainly the self-play wrapper)
    if args.is_footsies:
        if args.self_play.enabled:
            self_play_manager = SelfPlayManager(
                agent=None,
                env=env,
                max_opponents=args.self_play.max_opponents,
                snapshot_interval=args.self_play.snapshot_interval,
                switch_interval=args.self_play.switch_interval,
                mix_bot=args.self_play.mix_bot,
                log_dir=log_dir,
                starter_opponent=None, # default opponent is in-game bot initially
                evaluate_every=args.self_play.evaluate_every,
                csv_save=log_dir is not None,
            )

            if args.self_play.add_curriculum_opps:
                self_play_manager.populate_with_curriculum_opponents(
                    NSpammer(),
                    BSpammer(),
                    NSpecialSpammer(),
                    BSpecialSpammer(),
                    WhiffPunisher(),
                )

            LOGGER.info("Activated self-play")

            env = OpponentManagerWrapper(env, self_play_manager)
        
        elif args.curriculum.enabled:
            curriculum_manager = CurriculumManager(
                win_rate_threshold=args.curriculum.win_rate_threshold,
                win_rate_over_episodes=args.curriculum.win_rate_over_episodes,
                episode_threshold=args.curriculum.episode_threshold,
                log_dir=log_dir,
                csv_save=log_dir is not None,
            )

            LOGGER.info("Activated curriculum learning")

            env = OpponentManagerWrapper(env, curriculum_manager)

    return env


def setup_logger(agent_name: str, stdout_level: int = logging.INFO, file_level: int = logging.DEBUG, log_to_file: bool = True, multiprocessing: bool = False) -> logging.Logger:
    from logging.handlers import RotatingFileHandler
    from sys import stdout

    # Capture all warnings that are issued with the `warnings` package.
    # The advantage of `warnings` is that they are only issued once.
    logging.captureWarnings(True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    
    if multiprocessing:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    
    ch = logging.StreamHandler(stdout)
    ch.setFormatter(formatter)
    ch.setLevel(stdout_level)

    logger.addHandler(ch)

    if log_to_file:
        rfh = RotatingFileHandler(f"logs/{agent_name}.log", maxBytes=int(1e7), backupCount=9)
        rfh.setFormatter(formatter)
        rfh.setLevel(file_level)

        logger.addHandler(rfh)

    return logger


# For logging the win-rate of the SB3 algorithm
class WinRateCallback(BaseCallback):
    def __init__(self, log_frequency: int, log_dir: str, last: int = 100, start_step: int = 0):
        super().__init__()
        self.log_frequency = log_frequency
        self.summary_writer = SummaryWriter(log_dir=log_dir)
        self.current_step = start_step

        self._wins = deque([], maxlen=last)

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        terminated = self.locals["dones"][0]
        truncated = info["TimeLimit.truncated"]
        reward = self.locals["rewards"][0]

        if terminated or truncated:
            won = (info["guard"][0] > info["guard"][1]) if truncated else (reward > 0)
            self._wins.append(won)

        if self.current_step % self.log_frequency == 0:
            self.summary_writer.add_scalar(
                f"Performance/Win rate over the last {self._wins.maxlen} games",
                (sum(self._wins) / len(self._wins)) if self._wins else 0.5,
                self.current_step,
            )
        
        self.current_step += 1

        # Never stop because of this
        return True


def main(args: MainArgs):
    # Use the same logging directory as the one the environment uses. Everything should be logging to the same place.
    log_dir: str = args.log_folder

    # Alleviate the need of specifically specifying different ports for each parallel instance.
    # Still, allow the user to specify specific ports if they want to.
    ports = FootsiesEnv.find_ports(start=11000)
    args.env.kwargs.setdefault("game_port", ports["game_port"])
    args.env.kwargs.setdefault("opponent_port", ports["opponent_port"])
    args.env.kwargs.setdefault("remote_control_port", ports["remote_control_port"])

    # Set up the main logger

    setup_logger(args.agent.name, stdout_level=args.misc.log_stdout_level, file_level=args.misc.log_file_level, log_to_file=args.misc.log_file, multiprocessing=args.misc.hogwild)

    # Pre-processing

    if args.seed is not None:
        T.manual_seed(args.seed)
        random.seed(args.seed)
        LOGGER.info("Seed was set to %s", args.seed)

    # Prepare environment

    if LOGGER.isEnabledFor(logging.INFO):
        environment_initialization_msg = (
            f"Initializing {'FOOTSIES' if args.env.is_footsies else ('environment ' + args.env.name)}\n"
            " Environment arguments:\n"
        )
        environment_initialization_msg += "\n".join(f"  {k}: {v} ({type(v).__name__})" for k, v in args.env.kwargs.items())
        LOGGER.info(environment_initialization_msg)
    
    env = create_env(args.env, log_dir=log_dir)

    # Log which wrappers are being used
    e = env
    using_wrappers = "Using wrappers:"
    while not isinstance(e, FootsiesEnv):
        using_wrappers += f"\n {e.__class__.__name__}"
        e = e.env # type: ignore
    LOGGER.info(using_wrappers)

    # Prepare agent
    
    if args.agent.is_sb3:
        agent = import_sb3(args.agent.model, env, args.agent.kwargs)
            
    else:
        agent, loggables = import_agent(args.agent.model, env, args.agent.kwargs)

    if LOGGER.isEnabledFor(logging.INFO):
        agent_initialization_msg = (
            f"Imported agent '{args.agent.model + (' (SB3)' if args.agent.is_sb3 else '')}' with name '{args.agent.name}'\n"
            f" Agent arguments:\n"
        )
        agent_initialization_msg += "\n".join(f"  {k}: {v} ({type(v).__name__})" for k, v in args.agent.kwargs.items())
        LOGGER.info(agent_initialization_msg)

    if args.misc.load:
        load_agent(agent, args.agent.name)

    # Identity function, used when logging is disabled
    agent_logging_wrapper = lambda a: a
    if args.misc.log_tensorboard and not args.agent.is_sb3:
        LOGGER.info("Logging enabled on the agent")

        agent_logging_wrapper = lambda a: TrainingLoggerWrapper(
            a,
            log_frequency=args.misc.log_frequency,
            log_dir=log_dir,
            episode_reward=True,
            average_reward=True,
            average_reward_coef=0.99,
            win_rate=True,
            truncation=True,
            episode_length=True,
            test_states_number=args.misc.log_test_states_number,
            step_start_value=args.misc.log_step_start,
            episode_start_value=args.misc.log_episode_start,
            csv_save=True,
            **loggables, # type: ignore
        )
    
    if args.intrinsic_reward_scheme:
        intrinsic_reward_scheme = args.intrinsic_reward_scheme.basic()
        
    else:
        intrinsic_reward_scheme = None

    # We need to treat self-play differently, we can't set the snapshot method in create_env
    if args.env.self_play.enabled:
        if isinstance(agent, BaseAlgorithm):
            raise ValueError("it is not possible to use self-play with an SB3 agent")
        
        self_play_manager = extract_opponent_manager(env)

        if not isinstance(self_play_manager, SelfPlayManager):
            raise RuntimeError("even though it was requested, the environment does not have the self-play manager set, or it could not be found")
        
        self_play_manager.set_agent(agent)

    # Train

    if isinstance(agent, BaseAlgorithm):
        try:
            from stable_baselines3.common.logger import configure
            sb3_logger = configure(log_dir, ["tensorboard"])
            agent.set_logger(sb3_logger)

            if args.misc.log_tensorboard:
                logging_callback = WinRateCallback(
                    log_frequency=args.misc.log_frequency,
                    log_dir=log_dir,
                    start_step=args.misc.log_step_start,
                )
            else:
                logging_callback = None

            if args.time_steps is None:
                raise ValueError("the number of time steps has to be specified for SB3 algorithms")

            agent.learn(
                total_timesteps=args.time_steps,
                callback=logging_callback,
                tb_log_name=log_dir,
                reset_num_timesteps=False,
                progress_bar=True,
            )

        # NOTE: duplicated from train(...)
        except KeyboardInterrupt:
            LOGGER.info("Training manually interrupted")
        
        except Exception as e:
            LOGGER.exception(f"Training stopped due to {type(e).__name__}: '{e}', ignoring and quitting training")

    else:
        train_kwargs = {
            "n_episodes": args.episodes,
            "n_timesteps": args.time_steps,
            "penalize_truncation": args.penalize_truncation,
            "intrinsic_reward_scheme": intrinsic_reward_scheme,
            "progress_bar_kwargs": args.progress_bar_kwargs,
            "skip_freeze": args.skip_freeze,
            "initial_seed": args.seed,
        }

        if args.misc.hogwild:
            if args.episodes is None:
                raise NotImplementedError("indefinite training using Hogwild! is not supported (KeyboardInterrupt signal not handled by the main process), please set a fixed number of episodes for each worker to perform")
            if not isinstance(agent, FootsiesAgentTorch):
                raise ValueError("Hogwild! is only supported for PyTorch-based agents")
            
            # Close the dummy environment above, which was useful for setting up the agent.
            # However, each worker will have its own instance of the environment.
            env.close()

            # Define a wrapper on create_env, which instantiates a Gymnasium environment with the provided kwargs
            def create_env_with_args(env_args: EnvArgs, **kwargs):
                # I'm likely being paranoid with multiprocessing... just making sure *nothing* is shared
                env_args = deepcopy(env_args)

                # Indicate whether there are conflicting arguments before updating
                conflicting_kwargs = env_args.kwargs.keys() & kwargs.keys()
                if conflicting_kwargs:
                    LOGGER.warning(f"Will overwrite environment kwargs: {conflicting_kwargs}")
                
                env_args.kwargs.update(kwargs)

                return create_env(env_args)

            hogwild(
                agent,
                partial(create_env_with_args, env_args=args.env),
                train,
                n_workers=args.misc.hogwild_n_workers,
                cpus_to_use=args.misc.hogwild_cpus,
                is_footsies=args.env.is_footsies,
                logging_wrapper=agent_logging_wrapper,
                **train_kwargs,
            )
        
        else:
            agent = agent_logging_wrapper(agent)
            train(agent, env, **train_kwargs)

    if args.misc.save:
        save_agent(agent, args.agent.name)
        save_agent_parameters(args.agent.kwargs, args.agent.name)
        save_agent_training_args(asdict(args), args.agent.name)

    env.close()

    if isinstance(agent, TrainingLoggerWrapper):
        agent.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
