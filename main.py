import argparse
import os
import importlib
from footsies_gym.envs.footsies import FootsiesEnv
from agents.base import FootsiesAgentBase
from tqdm import tqdm
from itertools import count

"""
Practical considerations:

- Special attacks require holding an attack input without interruption for 1 whole second (60 frames).
  Therefore, the policy should ideally be able to consider a history at least 60 frames long.
"""

def main(agent: FootsiesAgentBase, game_path: str, n_episodes: int = None):
    env = FootsiesEnv(game_path=game_path, frame_delay=20)

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
    parser.add_argument("-m", "--model-kwargs", action="extend", nargs="+", type="str", help="key-value pairs to pass as keyword arguments to the agent implementation. Values are treated as floating-point numbers")

    args = parser.parse_args()

    if len(args.model_kwargs) % 2 != 0:
        raise ValueError("the values passed to '--model-kwargs' should be a list of key-value pairs")
    
    model_kwargs = {k: float(v) for k, v in zip(args.model_kwargs[0::2], args.model_kwargs[1::2])}

    agent_module_str = ".".join(("agents", args.agent, "agent"))
    agent_module = importlib.import_module(agent_module_str)
    agent = agent_module.FootsiesAgent(model_kwargs)

    main(agent, args.game_path, args.episodes)
