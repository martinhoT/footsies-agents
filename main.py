import argparse
import os
import importlib
import gymnasium as gym
import random
from gymnasium import Env
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.time_limit import TimeLimit
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.envs.exceptions import FootsiesGameClosedError
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.frame_skip import FootsiesFrameSkipped
from agents.base import FootsiesAgentBase
from tqdm import tqdm
from itertools import count
from typing import List, Any
from collections import deque
from stable_baselines3.common.base_class import BaseAlgorithm

from agents.logger import TrainingLoggerWrapper
from agents.utils import snapshot_sb3_policy, wrap_policy
from args import parse_args
from self_play import SelfPlayManager


"""
Practical considerations:

- Special attacks require holding an attack input without interruption for 1 whole second (60 frames).
  Therefore, the policy should ideally be able to consider a history at least 60 frames long.
"""

# TODO: try Optuna
# TODO: try adding noise to some observation variables (such as position) for better generalization?


def import_sb3(agent_name: str, env: Env, parameters: dict) -> BaseAlgorithm:
    import stable_baselines3
    agent_class = stable_baselines3.__dict__[agent_name.upper()]
    return agent_class(
        env=env,
        **parameters,
    )


def import_agent(agent_name: str, env: Env, parameters: dict) -> FootsiesAgentBase:
    agent_module_str = ".".join(("agents", agent_name, "agent"))
    agent_module = importlib.import_module(agent_module_str)
    return agent_module.FootsiesAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        **parameters,
    )


def import_loggables(agent_name: str, agent: FootsiesAgentBase) -> List[Any]:
    loggables_module_str = ".".join(("agents", agent_name, "loggables"))
    loggables_module = importlib.import_module(loggables_module_str)
    return loggables_module.get_loggables(agent)


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
        print("Agent loaded")

    else:
        print("Can't load agent, there was no agent saved!")


def save_agent_model(agent: FootsiesAgentBase | BaseAlgorithm, model_name: str, folder: str = "saved"):
    agent_folder_path = os.path.join(folder, model_name)
    is_footsies_agent = isinstance(agent, FootsiesAgentBase)

    if is_footsies_agent and not os.path.exists(agent_folder_path):
        os.makedirs(agent_folder_path)

    # Both FOOTSIES and SB3 agents use the same method and signature (mostly)
    agent.save(agent_folder_path)
    print("Agent saved")


def train(
    agent: FootsiesAgentBase,
    env: Env,
    n_episodes: int = None,
    penalize_truncation: float = None,
    self_play_manager: SelfPlayManager = None,
):
    """
    Train an `agent` on the given Gymnasium environment `env`

    Parameters
    ----------
    agent: FootsiesAgentBase
        implementation of the FootsiesAgentBase, representing the agent
    env: Env
        the Gymnasium environment to train on
    n_episodes: int
        if specified, the number of training episodes
    penalize_truncation: float
        penalize the agent if the time limit was exceeded, to discourage lengthening the episode
    self_play_manager: SelfPlayManager
        opponent pool manager for self-play. If None, self-play will not be performed
    """

    print("Preprocessing...", end=" ", flush=True)
    agent.preprocess(env)
    print("done!")

    training_iterator = count() if n_episodes is None else range(n_episodes)

    # Only used for self-play
    if self_play_manager is not None:
        self_play_manager._add_opponent(env.unwrapped.opponent)

    try:
        for _ in tqdm(training_iterator):
            obs, info = env.reset()

            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                
                if penalize_truncation is not None and truncated:
                    reward = penalize_truncation
                
                agent.update(obs, reward, terminated, truncated, info)

            # Set a new opponent from the opponent pool
            if self_play_manager is not None:
                if self_play_manager.update_at_episode():
                    env.unwrapped.set_opponent(self_play_manager.current_opponent)

    except KeyboardInterrupt:
        print("Training manually interrupted")

    except FootsiesGameClosedError:
        print("Game closed manually, quitting training")

    except Exception as e:
        print(
            f"Training stopped due to {type(e).__name__}: '{e}', ignoring and quitting training"
        )
        from traceback import print_exception

        print_exception(e)


if __name__ == "__main__":
    args = parse_args()

    # Prepare environment

    if args.env.is_footsies:
        print("Initializing FOOTSIES")

        # Set arguments so that training is easier by default
        env = FootsiesEnv(
            frame_delay=0,  # frame delay of 0 by default
            dense_reward=True,  # dense reward enabled by default
            **args.env.kwargs,
        )

        if args.env.footsies_wrapper_norm:
            print(" Adding FootsiesNormalized wrapper")
            env = FootsiesNormalized(env)

        if args.env.footsies_wrapper_fs:
            print(" Adding FootsiesFrameSkipped wrapper")
            env = FootsiesFrameSkipped(env)

        if args.env.wrapper_time_limit > 0:
            env = TimeLimit(env, max_episode_steps=args.env.wrapper_time_limit)

        env = FlattenObservation(env)

        if args.env.footsies_wrapper_acd:
            print(" Adding FootsiesActionCombinationsDiscretized wrapper")
            env = FootsiesActionCombinationsDiscretized(env)

    else:
        print(f"Initializing environment {args.env.name}")
        env = gym.make(args.env.name, **args.env.kwargs)

    print(" Environment arguments:")
    for k, v in args.env.kwargs.items():
        print(f"  {k}: {v} ({type(v).__name__})")

    # Prepare agent
    
    if args.agent.is_sb3:
        agent = import_sb3(args.agent.name, env, args.agent.kwargs)
            
    else:
        agent = import_agent(args.agent.name, env, args.agent.kwargs)

    print(f"Imported agent '{args.agent.name + (' (SB3)' if args.agent.is_sb3 else '')}' with name '{args.agent.model_name}'")
    print(f" Agent arguments:")
    for k, v in args.agent.kwargs.items():
        print(f"  {k}: {v} ({type(v).__name__})")

    if args.misc.load:
        load_agent_model(agent, args.agent.model_name)

    # Set a good default agent for self-play (FOOTSIES only)
    if args.self_play.enabled:
        footsies_env: FootsiesEnv = env.unwrapped
        if args.agent.is_sb3:
            footsies_env.set_opponent(wrap_policy(env, snapshot_sb3_policy(agent)))
        else:
            footsies_env.set_opponent(agent.extract_policy(env))

    if args.misc.log and not args.agent.is_sb3:
        print("Logging enabled")
        loggables = import_loggables(args.agent.name, agent)

        agent = TrainingLoggerWrapper(
            agent,
            log_frequency=args.misc.log_frequency,
            log_dir=args.misc.log_dir,
            cummulative_reward=True,
            win_rate=True,
            truncation=True,
            episode_length=True,
            test_states_number=args.misc.log_test_states_number,
            **loggables,
        )

    if args.agent.is_sb3:
        try:
            from stable_baselines3.common.logger import configure
            logger = configure(args.misc.log_dir, ["tensorboard"])
            agent.set_logger(logger)

            # opponent_pool = deque([], maxlen=args.self_play_max_snapshots)
            # if will_footsies_self_play:
            #     opponent_pool.append(env.unwrapped.opponent)

            agent.learn(
                total_timesteps=args.time_steps,
                tb_log_name=args.misc.log_dir,
                reset_num_timesteps=False,
                progress_bar=True,
            )

        # NOTE: duplicated from train(...)
        except KeyboardInterrupt:
            print("Training manually interrupted")
        
        except Exception as e:
            print(
                f"Training stopped due to {type(e).__name__}: '{e}', ignoring and quitting training"
            )
            from traceback import print_exception

            print_exception(e)

    else:
        train(
            agent,
            env,
            args.episodes,
            args.penalize_truncation,
            self_play_manager=SelfPlayManager(
                snapshot_method=(lambda: wrap_policy(env, snapshot_sb3_policy(agent))) if args.agent.is_sb3 else (lambda: agent.extract_policy(env)),
                snapshot_frequency=args.self_play.snapshot_freq,
                max_snapshots=args.self_play.max_snapshots,
                mix_bot=args.self_play.mix_bot,
            ),
        )

    if args.misc.save:
        save_agent_model(agent, args.agent.model_name)
