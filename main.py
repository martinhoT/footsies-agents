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
from agents.base import FootsiesAgentBase
from tqdm import tqdm
from itertools import count

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


def extract_kwargs(f_kwargs: dict, s_kwargs: dict, b_kwargs: dict) -> dict:
    kwargs = {}
    if f_kwargs is not None:
        if len(f_kwargs) % 2 != 0:
            raise ValueError(
                "the values passed to '--[...]-F-kwargs' should be a list of key-value pairs"
            )

        kwargs.update({k: float(v) for k, v in zip(f_kwargs[0::2], f_kwargs[1::2])})

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
    agent.preprocess(env)

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
        "-eF",
        "--env-F-kwargs",
        action="extend",
        nargs="+",
        type=str,
        help="key-value pairs to pass as keyword arguments to the environment. Values are treated as floating-point numbers",
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
        "-mF",
        "--model-F-kwargs",
        action="extend",
        nargs="+",
        type=str,
        help="key-value pairs to pass as keyword arguments to the agent implementation. Values are treated as floating-point numbers",
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

    args = parser.parse_args()

    env_kwargs = extract_kwargs(args.env_F_kwargs, args.env_S_kwargs, args.env_B_kwargs)
    model_kwargs = extract_kwargs(
        args.model_F_kwargs, args.model_S_kwargs, args.model_B_kwargs
    )

    if args.env == "FOOTSIES":
        if args.footsies_path is None:
            raise ValueError(
                "the path to the FOOTSIES executable should be specified with '--footsies-path' when using the FOOTSIES environment"
            )

        env = FootsiesEnv(
            game_path=args.footsies_path,
            frame_delay=0, # frame delay of 0 by default
            **env_kwargs,
        )

        if args.footsies_wrapper_norm:
            env = FootsiesNormalized(env)

        env = FlattenObservation(env)

        if args.footsies_wrapper_acd:
            env = FootsiesActionCombinationsDiscretized(env)

    else:
        env = gym.make(args.env)

    agent = import_agent(args.agent, env, model_kwargs)
    model_name = args.agent if args.model_name is None else args.model_name

    save = not args.no_save
    load = not args.no_load

    if load:
        load_agent_model(agent, model_name)

    train(agent, env, args.episodes)

    if save:
        save_agent_model(agent, model_name)
