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


"""
Practical considerations:

- Special attacks require holding an attack input without interruption for 1 whole second (60 frames).
  Therefore, the policy should ideally be able to consider a history at least 60 frames long.
"""

# TODO: try using Optuna


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


# Self-play assumes env is FootsiesEnv
def train(
    agent: FootsiesAgentBase,
    env: Env,
    n_episodes: int = None,
    self_play: bool = False,
    self_play_snapshot_frequency: int = 100,
    self_play_max_snapshots: int = 100,
    self_play_mix_bot: int = None,
    penalize_truncation: float = None,
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
    self_play: bool
        if true, use self-play during training. Assumes the environment is FootsiesEnv or a wrapped version of it
    self_play_snapshot_frequency: int
        how frequent to take a snapshot of the current policy for the opponent pool
    self_play_max_snapshots: int
        maximum capacity of the opponent pool. If at maximum, the oldest opponents are discarded
    self_play_mix_bot: int
        if specified, will include the in-game FOOTSIES bot as an opponent.
        Will enter after `self_play_mix_bot` episodes and stay for `self_play_mix_bot` episodes.
        As such, the opponent distribution will be 50/50, distributed between the snapshots and the in-game bot.
        This argument merely controls the switch frequency
    penalize_truncation: float
        penalize the agent if the time limit was exceeded, to discourage lengthening the episode
    """

    print("Preprocessing...", end=" ", flush=True)
    agent.preprocess(env)
    print("done!")

    training_iterator = count() if n_episodes is None else range(n_episodes)

    # Only used for self-play
    opponent_pool = deque([], maxlen=self_play_max_snapshots)
    if self_play:
        opponent_pool.append(env.unwrapped.opponent)

    mix_bot_counter = 0
    mix_bot_playing = False

    try:
        for episode in tqdm(training_iterator):
            obs, info = env.reset()

            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                
                if penalize_truncation is not None and truncated:
                    reward = penalize_truncation
                
                agent.update(obs, reward, terminated, truncated)

            # Set a new opponent from the opponent pool
            if self_play:
                # Perform a snapshot of the agent at the current
                if episode % self_play_snapshot_frequency == 0:
                    print("Agent snapshot created!")
                    opponent_pool.append(agent.extract_policy(env))

                mix_bot_counter += 1
                # Switch to the bot if the counter has surpassed the threshold
                if self_play_mix_bot is not None and mix_bot_counter >= self_play_mix_bot:
                    # Go back to using opponent pool opponents
                    if mix_bot_playing:
                        mix_bot_counter = 0
                        mix_bot_playing = False
                        print("Will use opponents from the opponent pool now!")
                
                    # Start using the in-game bot instead
                    else:
                        env.unwrapped.set_opponent(None)
                        mix_bot_counter = 0
                        mix_bot_playing = True
                        print("Will use the in-game opponent now!")
                
                # As long as the in-game bot is not playing, we will switch opponent every game
                if not mix_bot_playing:
                    print("Switched to new opponent from opponent pool!")
                    new_opponent = random.sample(opponent_pool, 1)[0]
                    env.unwrapped.set_opponent(new_opponent)
            
            # NOTE: necessary so that the FOOTSIES environment can restart on outside truncation
            if truncated and isinstance(env.unwrapped, FootsiesEnv):
                print("Environment truncated!")
                env.unwrapped.hard_reset()

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
        help="use self-play during training on the FOOTSIES environment. It's recommended to use the time limit wrapper. Note: SB3 agents don't support this feature",
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

    args = parser.parse_args()

    # Prepare various variables, including keyword arguments

    env_kwargs = extract_kwargs(args.env_N_kwargs, args.env_S_kwargs, args.env_B_kwargs)
    model_kwargs = extract_kwargs(
        args.model_N_kwargs, args.model_S_kwargs, args.model_B_kwargs
    )

    is_sb3 = args.agent.startswith("sb3.")
    will_footsies_self_play = args.footsies_self_play and args.env == "FOOTSIES"

    if is_sb3:
        if will_footsies_self_play:
            print("WARN: self-play with SB3 algorithms is not supported, self-play will be disabled")
            will_footsies_self_play = False
        
        if args.episodes is not None:
            print("WARN: specifying a number of episodes for SB3 algorithms is not supported, will be ignored")

    if will_footsies_self_play:
        # Set dummy opponent for now, and set later with a copy of the instanced agent
        env_kwargs["opponent"] = lambda o: (False, False, False)
        env_kwargs["opponent_port"] = args.footsies_self_play_port

    # Prepare environment

    if args.env == "FOOTSIES":
        print("Initializing FOOTSIES")
        if args.footsies_path is None:
            raise ValueError(
                "the path to the FOOTSIES executable should be specified with '--footsies-path' when using the FOOTSIES environment"
            )

        # Set arguments so that training is easier by default
        env = FootsiesEnv(
            game_path=args.footsies_path,
            frame_delay=0,  # frame delay of 0 by default
            dense_reward=True,  # dense reward enabled by default
            **env_kwargs,
        )

        if args.footsies_wrapper_norm:
            print(" Adding FootsiesNormalized wrapper")
            env = FootsiesNormalized(env)

        if args.footsies_wrapper_fs:
            print(" Adding FootsiesFrameSkipped wrapper")
            env = FootsiesFrameSkipped(env)

        if args.wrapper_time_limit > 0:
            env = TimeLimit(env, max_episode_steps=args.wrapper_time_limit)

        env = FlattenObservation(env)

        if args.footsies_wrapper_acd:
            print(" Adding FootsiesActionCombinationsDiscretized wrapper")
            env = FootsiesActionCombinationsDiscretized(env)

    else:
        print(f"Initializing environment {args.env}")
        env = gym.make(args.env, **env_kwargs)

    print(" Environment arguments:")
    for k, v in env_kwargs.items():
        print(f"  {k}: {v} ({type(v).__name__})")

    # Prepare agent
    
    if is_sb3:
        agent_name = args.agent[4:]
        agent = import_sb3(agent_name, env, model_kwargs)
            
    else:
        agent_name = args.agent
        agent = import_agent(agent_name, env, model_kwargs)

    model_name = agent_name if args.model_name is None else args.model_name

    print(f"Imported agent '{agent_name + (' (SB3)' if is_sb3 else '')}' with name '{model_name}'")
    print(f" Agent arguments:")
    for k, v in model_kwargs.items():
        print(f"  {k}: {v} ({type(v).__name__})")

    save = not args.no_save
    load = not args.no_load

    if load:
        load_agent_model(agent, model_name)

    # Set a good default agent for self-play
    if will_footsies_self_play:
        footsies_env: FootsiesEnv = env.unwrapped
        footsies_env.set_opponent(agent.extract_policy(env))

    if not args.no_log and not is_sb3:
        print("Logging enabled")
        loggables = import_loggables(args.agent, agent)

        agent = TrainingLoggerWrapper(
            agent,
            log_frequency=args.log_frequency,
            log_dir=args.log_dir,
            cummulative_reward=True,
            win_rate=True,
            truncation=True,
            episode_length=True,
            test_states_number=args.log_test_states_number,
            **loggables,
        )

    if is_sb3:
        try:
            agent.learn(
                total_timesteps=args.time_steps,
                tb_log_name=args.log_dir,
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
            will_footsies_self_play,
            args.footsies_self_play_snapshot_freq,
            args.footsies_self_play_max_snapshots,
            args.footsies_self_play_mix_bot,
            args.penalize_truncation,
        )

    if save:
        save_agent_model(agent, model_name)
