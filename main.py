import argparse
import os
import importlib
from footsies_gym.envs.footsies import FootsiesEnv
from agents.base import FootsiesAgentBase
from tqdm import tqdm
from itertools import count


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

    args = parser.parse_args()

    agent_module_str = ".".join(("agents", args.agent, "agent"))
    agent_module = importlib.import_module(agent_module_str)
    agent = agent_module.FootsiesAgent()

    main(agent, args.game_path, args.episodes)
