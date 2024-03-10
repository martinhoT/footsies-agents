import os
import importlib
import gymnasium as gym
import torch
import logging
from gymnasium import Env
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from gymnasium.wrappers.time_limit import TimeLimit
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.envs.exceptions import FootsiesGameClosedError
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.frame_skip import FootsiesFrameSkipped
from tqdm import tqdm
from itertools import count
from copy import deepcopy
from functools import partial
from typing import List, Any, Callable
from stable_baselines3.common.base_class import BaseAlgorithm
from agents.base import FootsiesAgentBase, FootsiesAgentTorch
from agents.diayn import DIAYN, DIAYNWrapper
from agents.logger import TrainingLoggerWrapper
from agents.utils import snapshot_sb3_policy, wrap_policy
from agents.torch_utils import hogwild
from args import parse_args, EnvArgs
from self_play import SelfPlayManager

LOGGER = logging.getLogger("main")


"""
Practical considerations:

- Special attacks require holding an attack input without interruption for 1 whole second (60 frames).
  Therefore, the policy should ideally be able to consider a history at least 60 frames long.
"""

# TODO: try adding noise to some observation variables (such as position) for better generalization?


def import_sb3(agent_model: str, env: Env, parameters: dict) -> BaseAlgorithm:
    import stable_baselines3
    agent_class = stable_baselines3.__dict__[agent_model.upper()]
    return agent_class(
        policy="MlpPolicy", # we assume we will never need the CnnPolicy
        env=env,
        **parameters,
    )


def import_agent(agent_model: str, env: Env, parameters: dict) -> tuple[FootsiesAgentBase, dict[str, list]]:
    model_init_module_str = ".".join(("models", agent_model))
    model_init_module = importlib.import_module(model_init_module_str)

    agent, loggables = model_init_module.model_init(observation_space_size=env.observation_space.shape[0], action_space_size=env.action_space.n, **parameters)

    return agent, loggables


def load_agent_model(agent: FootsiesAgentBase | BaseAlgorithm, model_name: str, folder: str = "saved"):
    agent_folder_path = os.path.join(folder, model_name)
    is_footsies_agent = isinstance(agent, FootsiesAgentBase)
    if not is_footsies_agent:
        agent_folder_path = agent_folder_path + ".zip"

    if os.path.exists(agent_folder_path):
        if is_footsies_agent and not os.path.isdir(agent_folder_path):
            raise OSError(f"the existing file '{agent_folder_path}' is not a folder!")

        if is_footsies_agent:
            agent.load(agent_folder_path)
        else:
            agent.set_parameters(agent_folder_path)
        LOGGER.info("Agent '%s' loaded", model_name)

    else:
        LOGGER.info("Can't load agent '%s', there was no agent saved!", model_name)


def save_agent_model(agent: FootsiesAgentBase | BaseAlgorithm, model_name: str, folder: str = "saved"):
    agent_folder_path = os.path.join(folder, model_name)
    is_footsies_agent = isinstance(agent, FootsiesAgentBase)

    if is_footsies_agent and not os.path.exists(agent_folder_path):
        os.makedirs(agent_folder_path)

    # Both FOOTSIES and SB3 agents use the same method and signature (mostly)
    agent.save(agent_folder_path)
    LOGGER.info("Agent '%s' saved", model_name)


def train(
    agent: FootsiesAgentBase,
    env: Env,
    n_episodes: int = None,
    penalize_truncation: float = None,
    self_play_manager: SelfPlayManager = None,
    episode_finished_callback: Callable[[int], None] = lambda episode: None,
    progress_bar: bool = True,
):
    """
    Train an `agent` on the given Gymnasium environment `env`.

    Parameters
    ----------
    - `agent`: implementation of FootsiesAgentBase, representing the agent
    - `env`: the Gymnasium environment to train on
    - `n_episodes`: if specified, the number of training episodes
    - `penalize_truncation`: penalize the agent if the time limit was exceeded, to discourage lengthening the episode
    - `self_play_manager`: opponent pool manager for self-play. If None, self-play will not be performed
    - `episode_finished_callback`: function that will be called after each episode is finished
    - `progress_bar`: whether to display a progress bar (with `tqdm`)
    """
    agent.preprocess(env)
    LOGGER.info("Preprocessing done!")

    training_iterator = count() if n_episodes is None else range(n_episodes)

    # Only used for self-play
    if self_play_manager is not None:
        self_play_manager._add_opponent(env.unwrapped.opponent)

    if progress_bar:
        training_iterator = tqdm(training_iterator)

    try:
        for episode in training_iterator:
            obs, info = env.reset()

            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = agent.act(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                
                if penalize_truncation is not None and truncated:
                    reward = penalize_truncation
                
                agent.update(obs, reward, terminated, truncated, info)

            # Set a new opponent from the opponent pool
            if self_play_manager is not None:
                should_change = self_play_manager.update_at_episode()
                if should_change:
                    env.unwrapped.set_opponent(self_play_manager.current_opponent)
            
            episode_finished_callback(episode)

    except KeyboardInterrupt:
        LOGGER.info("Training manually interrupted")

    except FootsiesGameClosedError as e:
        LOGGER.warning("Quitting training since game closed: '%s'", e)

    except Exception as e:
        LOGGER.exception("Training stopped due to %s: '%s', ignoring and quitting training", type(e).__name__, e)


def create_env(args: EnvArgs) -> Env:
    # Create environment with initial wrappers
    if args.is_footsies:
        env = FootsiesEnv(**args.kwargs)

        if args.footsies_wrapper_norm:
            env = FootsiesNormalized(env)

        if args.footsies_wrapper_fs:
            env = FootsiesFrameSkipped(env)

    else:
        env = gym.make(args.name, **args.kwargs)

    # Wrap with additional, environment-independent wrappers
    if args.wrapper_time_limit > 0:
        env = TimeLimit(env, max_episode_steps=args.wrapper_time_limit)

    env = FlattenObservation(env)

    # Final FOOTSIES wrappers
    if args.is_footsies:
        if args.footsies_wrapper_acd:
            env = FootsiesActionCombinationsDiscretized(env)

    if args.torch:
        env = TransformObservation(env, lambda obs: torch.from_numpy(obs).float().unsqueeze(0))

    # Final miscellaneous wrappers
    if args.diayn:
        env = DIAYNWrapper(
            env,
            DIAYN(
                observation_dim=env.observation_space.shape[0],
                skill_dim=args.diayn_kwargs["skill_dim"],
                include_baseline=args.diayn_kwargs["include_baseline"],
                discriminator_learning_rate=args.diayn_kwargs["discriminator_learning_rate"],
                discriminator_hidden_layer_sizes=args.diayn_kwargs["discriminator_hidden_layer_sizes"],
                discriminator_hidden_layer_activation=args.diayn_kwargs["discriminator_hidden_layer_activation"],
            ),
            log_dir=args.diayn_kwargs["log_dir"],
            log_frequency=args.diayn_kwargs["log_frequency"],
        )

    return env


def setup_logger(agent_name: str, level: int = logging.DEBUG, log_to_file: bool = True, multiprocessing: bool = False) -> logging.Logger:
    from logging.handlers import RotatingFileHandler
    from sys import stdout

    logger = logging.getLogger("main")
    logger.setLevel(level)

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
    ch.setLevel(logging.WARNING)

    logger.addHandler(ch)

    if log_to_file:
        rfh = RotatingFileHandler(f"logs/{agent_name}.log", maxBytes=1e7, backupCount=9)
        rfh.setFormatter(formatter)
        rfh.setLevel(level)

        logger.addHandler(rfh)

    return logger


if __name__ == "__main__":
    args = parse_args()
    # Perform some argument post-processing
    log_dir = os.path.join("runs", args.agent.name)
    args.env.diayn_kwargs["log_dir"] = log_dir
    
    setup_logger(args.agent.name, log_to_file=args.misc.log, multiprocessing=args.misc.hogwild)

    # Prepare environment

    if LOGGER.isEnabledFor(logging.INFO):
        environment_initialization_msg = (
            f"Initializing {'FOOTSIES' if args.env.is_footsies else ('environment ' + args.env.name)}\n"
            " Environment arguments:\n"
        )
        environment_initialization_msg += "\n".join(f"  {k}: {v} ({type(v).__name__})" for k, v in args.env.kwargs.items())
        LOGGER.info(environment_initialization_msg)
    
    env = create_env(args.env)

    # Log which wrappers are being used
    e = env
    using_wrappers = "Using wrappers:"
    while not isinstance(e, FootsiesEnv):
        using_wrappers += f"\n {e.__class__.__name__}"
        e = e.env
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
        load_agent_model(agent, args.agent.name)

    # Set a good default agent for self-play (FOOTSIES only)
    if args.self_play.enabled:
        footsies_env: FootsiesEnv = env.unwrapped
        if args.agent.is_sb3:
            footsies_env.set_opponent(wrap_policy(env, snapshot_sb3_policy(agent)))
        else:
            footsies_env.set_opponent(agent.extract_policy(env))

    # Identity function, used when logging is disabled
    agent_logging_wrapper = lambda a: a
    if args.misc.log and not args.agent.is_sb3:
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
            **loggables,
        )

    if args.agent.is_sb3:
        try:
            from stable_baselines3.common.logger import configure
            sb3_logger = configure(log_dir, ["tensorboard"])
            agent.set_logger(sb3_logger)

            # opponent_pool = deque([], maxlen=args.self_play_max_snapshots)
            # if will_footsies_self_play:
            #     opponent_pool.append(env.unwrapped.opponent)

            agent.learn(
                total_timesteps=args.time_steps,
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
        self_play_manager = SelfPlayManager(
            snapshot_method=(lambda: wrap_policy(env, snapshot_sb3_policy(agent))) if args.agent.is_sb3 else (lambda: agent.extract_policy(env)),
            snapshot_frequency=args.self_play.snapshot_freq,
            max_snapshots=args.self_play.max_snapshots,
            mix_bot=args.self_play.mix_bot,
        ) if args.self_play.enabled else None

        train_kwargs = {
            "n_episodes": args.episodes,
            "penalize_truncation": args.penalize_truncation,
            "self_play_manager": self_play_manager,
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
        save_agent_model(agent, args.agent.name)

    env.close()
