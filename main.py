import argparse
import os
import importlib
import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers.flatten_observation import FlattenObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.envs.exceptions import FootsiesGameClosedError
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.frame_skip import FootsiesFrameSkipped
from agents.base import FootsiesAgentBase
from tqdm import tqdm
from itertools import count
from typing import List, Any

from agents.logger import TrainingLoggerWrapper

"""
Practical considerations:

- Special attacks require holding an attack input without interruption for 1 whole second (60 frames).
  Therefore, the policy should ideally be able to consider a history at least 60 frames long.
"""


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


def load_agent_model(agent: FootsiesAgentBase, model_name: str, folder: str = "saved"):
    agent_folder_path = os.path.join(folder, model_name)

    if os.path.exists(agent_folder_path):
        if not os.path.isdir(agent_folder_path):
            raise OSError(f"the existing file {agent_folder_path} is not a folder!")

        agent.load(agent_folder_path)
        print("Agent loaded")

    else:
        print("Can't load agent, there was no agent saved!")


def save_agent_model(agent: FootsiesAgentBase, model_name: str, folder: str = "saved"):
    agent_folder_path = os.path.join(folder, model_name)

    if not os.path.exists(agent_folder_path):
        os.makedirs(agent_folder_path)

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

        kwargs.update({k: convert_to_int_or_float(v) for k, v in zip(n_kwargs[0::2], n_kwargs[1::2])})

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


def train(
    agent: FootsiesAgentBase,
    env: Env,
    n_episodes: int = None,
):
    print("Preprocessing...", end=" ", flush=True)
    agent.preprocess(env)
    print("done!")

    training_iterator = count() if n_episodes is None else range(n_episodes)

    try:
        for episode in tqdm(training_iterator):
            obs, info = env.reset()

            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                agent.update(obs, reward, terminated, truncated)

    except KeyboardInterrupt:
        print("Training manually interrupted")

    except FootsiesGameClosedError:
        print("Game closed manually, quitting training")
    
    except Exception as e:
        print(f"Training stopped due to {type(e).__name__}: '{e}', ignoring and quitting training")
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
        help=f"agent implementation to use (available: {available_agents_str})",
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
        help="use the Frame Skipped wrapper for FOOTSIES. Only has an effect when using the FOOTSIES environment"
    )
    parser.add_argument("--episodes", type=int, default=None, help="number of episodes")
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
        "--log-frequency", type=int, default=5000, help="number of time steps between each log"
    )
    parser.add_argument(
        "--log-test-states-number", type=int, default=5000, help="number of test states to use when evaluating some metrics for logging"
    )
    parser.add_argument(
        "--log-dir", type=str, default=None, help="directory to which Tensorboard logs will be written"
    )

    args = parser.parse_args()

    env_kwargs = extract_kwargs(args.env_N_kwargs, args.env_S_kwargs, args.env_B_kwargs)
    model_kwargs = extract_kwargs(
        args.model_N_kwargs, args.model_S_kwargs, args.model_B_kwargs
    )

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

    print(f"Importing agent '{args.agent}'")
    print(f" Agent arguments:")
    for k, v in model_kwargs.items():
        print(f"  {k}: {v} ({type(v).__name__})")

    agent = import_agent(args.agent, env, model_kwargs)
    model_name = args.agent if args.model_name is None else args.model_name

    save = not args.no_save
    load = not args.no_load

    if load:
        load_agent_model(agent, model_name)

    if not args.no_log:
        print("Logging enabled")
        loggables = import_loggables(args.agent, agent)

        agent = TrainingLoggerWrapper(
            agent,
            log_frequency=args.log_frequency,
            log_dir=args.log_dir,
            cummulative_reward=True,
            win_rate=True,
            test_states_number=args.log_test_states_number,
            **loggables,
        )

    train(agent, env, args.episodes)

    if save:
        save_agent_model(agent, model_name)
