import argparse
import os
import importlib
from gymnasium import Env
from gymnasium.wrappers.flatten_observation import FlattenObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.move_frame_norm import FootsiesMoveFrameNormalized
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from agents.base import FootsiesAgentBase
from tqdm import tqdm
from itertools import count

"""
Practical considerations:

- Special attacks require holding an attack input without interruption for 1 whole second (60 frames).
  Therefore, the policy should ideally be able to consider a history at least 60 frames long.
"""


def main(agent: FootsiesAgentBase, env: Env, n_episodes: int = None):
    training_iterator = count() if n_episodes is None else range(n_episodes)

    try:
        for episode in tqdm(training_iterator):
            obs, info = env.reset()

            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                agent.update(obs, reward)

    except KeyboardInterrupt:
        print("Training manually interrupted")


if __name__ == "__main__":
    available_agents = [file.name for file in os.scandir("agents") if file.is_dir()]
    available_agents_str = ", ".join(available_agents)

    default_str = " (default: %(default)s)"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "agent",
        type=str,
        help=f"agent implementation to use (available: {available_agents_str})",
    )
    parser.add_argument(
        "game_path", type=str, help="location of the FOOTSIES executable"
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=None,
        help="number of episodes" + default_str,
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

    model_kwargs = {}
    if args.model_F_kwargs is not None:
        if len(args.model_F_kwargs) % 2 != 0:
            raise ValueError(
                "the values passed to '--model-F-kwargs' should be a list of key-value pairs"
            )

        model_kwargs.update({
            k: float(v)
            for k, v in zip(args.model_F_kwargs[0::2], args.model_F_kwargs[1::2])
        })
    
    if args.model_S_kwargs is not None:
        if len(args.model_S_kwargs) % 2 != 0:
            raise ValueError(
                "the values passed to '--model-S-kwargs' should be a list of key-value pairs"
            )

        model_kwargs.update(dict(zip(args.model_kwargs[0::2], args.model_kwargs[1::2])))
    
    if args.model_B_kwargs is not None:
        if len(args.model_B_kwargs) % 2 != 0:
            raise ValueError(
                "the values passed to '--model-B-kwargs' should be a list of key-value pairs"
            )
        
        for k, v in zip(args.model_B_kwargs[0::2], args.model_B_kwargs[1::2]):
            v_lower = v.lower()
            if v_lower == "true":
                model_kwargs[k] = True
            elif v_lower == "false":
                model_kwargs[k] = False
            else:
                raise ValueError(f"the value passed to key '{k}' on the '--model-B-kwargs' kwarg list is not a boolean ('{v}' is not 'true' or 'false')")
        
    env = FootsiesActionCombinationsDiscretized(
        FlattenObservation(
            FootsiesMoveFrameNormalized(
                FootsiesEnv(game_path=args.game_path, frame_delay=0)
            )
        )
    )

    agent_module_str = ".".join(("agents", args.agent, "agent"))
    agent_module = importlib.import_module(agent_module_str)
    agent = agent_module.FootsiesAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        **model_kwargs,
    )

    main(agent, env, args.episodes)
