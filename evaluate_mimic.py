import os
from agents.logger import TrainingLoggerWrapper
from main import import_loggables, load_agent_model, train
from gymnasium.wrappers.flatten_observation import FlattenObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.statistics import FootsiesStatistics
from agents.mimic.agent import FootsiesAgent as MimicAgent
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

agent = MimicAgent(env.observation_space, env.action_space,
    by_primitive_actions=False,
    optimize_frequency=3,
    use_sigmoid=True,
)

load_agent_model(agent, "mimic")



fig, ax = plt.subplots()
ax: Axes
line, = ax.plot([], [], lw=2)
ax.grid()
xdata, ydata = [], []

def init():
    ax.set_title("Win rate")
    ax.set_ylim(ymin=-0.1, ymax=1.1)
    ax.set_xlim(1, 10)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,

def interact():
    wins = 0
    games = 0

    while True:
        obs, info = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.act(obs, p1=False, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
        
        games += 1
        wins += 1 if reward > 0 else 0
        yield games, wins / games

def run(data):
    print(data)
    # Update the data
    t, y = data
    xdata.append(t)
    ydata.append(y)
    xmin, xmax = ax.get_xlim()

    if t > xmax:
        ax.set_xlim(xmin, xmax + 10)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)

    return line,

ani = animation.FuncAnimation(fig, run, interact, init_func=init, save_count=100)
plt.show()
