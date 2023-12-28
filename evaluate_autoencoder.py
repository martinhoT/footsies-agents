import os
from agents.logger import TrainingLoggerWrapper
from main import import_loggables, load_agent_model, train
from gymnasium.wrappers.flatten_observation import FlattenObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.statistics import FootsiesStatistics
from agents.autoencoder.agent import FootsiesAgent as AutoencoderAgent
import pprint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.axes import Axes

pp = pprint.PrettyPrinter(indent=4)

footsies_env = FootsiesEnv(
    game_path="../Footsies-Gym/Build/FOOTSIES.x86_64",
    render_mode="human",
    sync_mode="synced_non_blocking",
    game_port=15000,
    opponent_port=15001,
    log_file=os.path.join(os.getcwd(), "out.log"),
    log_file_overwrite=True,
    by_example=True,
    fast_forward=False,
)

statistics = FootsiesStatistics(footsies_env)

env = FootsiesActionCombinationsDiscretized(
    FlattenObservation(
        FootsiesNormalized(statistics)
    )
)

agent = AutoencoderAgent(env.observation_space, env.action_space,
    encoded_dim=3,
    normalized=True,
    include_sequentiality_loss=False,
)

load_agent_model(agent, "autoencoder")



fig = plt.figure()
ax: Axes = fig.add_subplot(projection="3d")
quiver = ax.quiver([0], [0], [0], [0], [0], [0])

def init():
    ax.set_title("Encoded state")
    quiver.set_UVC([0], [0], [0])
    return quiver

def interact():
    while True:
        obs, info = env.reset()
        terminated, truncated = False, False
        yield agent.encode(obs)

        while not (terminated or truncated):
            action = env.action_space.sample()   # doesn't matter
            obs, reward, terminated, truncated, info = env.step(action)
            yield agent.encode(obs)

def run(data):
    print(data)
    # Update the data
    quiver.set_UVC(*([c] for c in data))
    return quiver

ani = animation.FuncAnimation(fig, run, interact, init_func=init, save_count=100)
plt.show()
