from itertools import cycle
from gymnasium import Env
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from agents.base import FootsiesAgentBase
from typing import List, Callable, Any, Tuple
from footsies_gym.envs.footsies import FootsiesEnv


class TrainingLoggerWrapper(FootsiesAgentBase):
    def __init__(
        self,
        agent: FootsiesAgentBase,
        log_frequency: int,
        cummulative_reward: bool = False,
        win_rate: bool = False,
        network_histograms: List[nn.Module] = None,
        custom_evaluators: List[Tuple[str, Callable[[], float]]] = None,
        custom_evaluators_over_test_states: List[
            Tuple[str, Callable[[List[Tuple[Any, Any]]], float]]
        ] = None,
        test_states_number: int = 10_000,
        log_dir: str = None,
    ):
        """
        A wrapper on FOOTSIES agents which logs the specified training metrics.

        Parameters
        ----------
        agent: FootsiesAgentBase
            the agent that will be wrapped
        log_frequency: int
            how frequent, in terms of time steps, should logs be written

        cummulative_reward: bool
            the total accumulated reward since training started
        win_rate: bool
            the agent's win rate. Only makes sense for the FOOTSIES environment
        network_histograms: List[torch.nn.Module]
            list of networks whose weight and bias histograms will be logged
        custom_evaluators: List[Tuple[str, Callable[[], float]]]
            methods of the underlying agent whose results will be logged.
            It's a list of tuples where the first element is the metric's title/tag and the second is the method
        custom_evaluators_over_test_states: List[Tuple[str, Callable[[List[Tuple[Any, Any]]], float]]]
            methods of the underlying agent that accept a set of test state-action pairs as input and print a metric to be logged.
            The list of state-action pairs won't change in future calls of those methods, so the agents can cache it if they have to perform some transformations on it.
            If there is at least one evaluator, then `preprocess` will collect those test state-action pairs.
            It's a list with the same structure as `custom_evaluators`
        """
        self.agent = agent
        self.log_frequency = log_frequency
        self.cummulative_reward = cummulative_reward
        self.win_rate = win_rate
        self.network_histograms = [] if network_histograms is None else network_histograms
        self.custom_evaluators = [] if custom_evaluators is None else custom_evaluators
        self.custom_evaluators_over_test_states = (
            []
            if custom_evaluators_over_test_states is None
            else custom_evaluators_over_test_states
        )

        # In case there are custom evaluators over test states
        self.test_states = []
        self.test_states_number = test_states_number

        self.summary_writer = SummaryWriter(log_dir=log_dir)
        self.cummulative_reward = 0
        self.current_step = 0
        self.current_episode = 0

    def act(self, obs) -> "any":
        return self.agent.act(obs)

    def update(self, next_obs, reward: float, terminated: bool, truncated: bool):
        self.agent.update(next_obs, reward, terminated, truncated)

        self.cummulative_reward += reward
        self.current_step += 1

        if terminated or truncated:
            self.current_episode += 1

        # Write logs
        if self.current_step % self.log_frequency == 0:
            if self.cummulative_reward:
                self.summary_writer.add_scalar(
                    "Cummulative reward", self.cummulative_reward, self.current_step
                )
            if self.win_rate:
                self.summary_writer.add_scalar(
                    "Win rate",
                    (self.current_episode + self.cummulative_reward)
                    / (2 * self.current_episode)
                    if self.current_episode >= 1
                    else 0.5,
                    self.current_step,
                )

            for network in self.network_histograms:
                for layer_name, layer in enumerate(network.named_parameters()):
                    self.summary_writer.add_histogram(layer_name, layer, self.current_step)

            for tag, evaluator in self.custom_evaluators:
                self.summary_writer.add_scalar(tag, evaluator(), self.current_step)

            for tag, evaluator in self.custom_evaluators_over_test_states:
                self.summary_writer.add_scalar(tag, evaluator(self.test_states), self.current_step)

    def preprocess(self, env: Env):
        self.agent.preprocess(env)

        if len(self.custom_evaluators_over_test_states) > 0:
            state, _ = env.reset()

            for _ in range(self.test_states_number):
                action = env.action_space.sample()
                self.test_states.append((state, action))
                state, _, terminated, truncated, _ = env.step(action)

                if terminated or truncated:
                    env.reset()
            
            self.test_states.append((state, None))
            
            if isinstance(env.unwrapped, FootsiesEnv):
                env.unwrapped.hard_reset()
            else:
                env.close()

    def load(self, folder_path: str):
        self.agent.load(folder_path)

    def save(self, folder_path: str):
        self.agent.save(folder_path)
