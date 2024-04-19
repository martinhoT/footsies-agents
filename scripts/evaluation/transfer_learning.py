# %% Make sure we are running in the project's root

from os import chdir
chdir("/home/martinho/projects/footsies-agents")

# %% Imports

from main import load_agent, load_agent_parameters
from agents.ql.ql import QFunctionNetwork
from models import to_
from copy import deepcopy
from os import path
from scripts.evaluation.utils import create_env, WinRateObserver, plot_data, get_data_custom_loop

# %% Prepare environment

dummy_env, _ = create_env()

# %% Import base agent

# Learnt against the in-game bot
AGENT_NAME = "REF"

agent_parameters = load_agent_parameters(AGENT_NAME)
agent, _ = to_(
    observation_space_size=dummy_env.observation_space.shape[0],
    action_space_size=dummy_env.action_space.n,
    **agent_parameters
)
load_agent(agent, AGENT_NAME)

# %% Extract critic

critic: QFunctionNetwork = agent.a2c.learner.critic

# %% Create new agents from scratch, but one of them uses the already-made Q-function network

agent_0, _ = to_(
    observation_space_size=dummy_env.observation_space.shape[0],
    action_space_size=dummy_env.action_space.n,
    **agent_parameters
)

agent_1 = deepcopy(agent_0)

agent_1_critic: QFunctionNetwork = agent_1.a2c.learner.critic
agent_1_critic.q_network.load_state_dict(critic.q_network.state_dict())

agents = [
    ("control", agent_0),
    ("initted", agent_1),
]

opponent = agent.extract_opponent(dummy_env)

# %% Get data, if possible

result_path = path.splitext(__file__)[0] 

dfs = get_data_custom_loop(result_path, agents, WinRateObserver, opponent, seeds=10, episodes=1000)
if dfs is None:
    print("Could not get data, quitting")
    exit(0)

# %% Plot results

plot_data(
    dfs=dfs,
    title="Win rate over the last 100 episodes",
    fig_path=result_path,
    exp_factor=0.9,
    xlabel="Episode",
    ylabel="Win rate",
    run_name_mapping={
        "control": "No learned Q-function",
        "initted": "With learned Q-function",
    },
)
