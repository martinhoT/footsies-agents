from gymnasium import Env
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from agents.base import FootsiesAgentBase
from typing import List, Callable, Any, Tuple
from collections import deque
import logging

LOGGER = logging.getLogger("main.tensorboard")


class TrainingLoggerWrapper(FootsiesAgentBase):
    def __init__(
        self,
        agent: FootsiesAgentBase,
        log_frequency: int,
        log_dir: str = None,
        episode_reward: bool = False,
        average_reward: bool = False,
        average_reward_coef: float = None,
        win_rate: bool = False,
        truncation: bool = False,
        episode_length: bool = False,
        network_histograms: List[nn.Module] = None,
        custom_evaluators: List[Tuple[str, Callable[[], float]]] = None,
        custom_evaluators_over_test_states: List[
            Tuple[str, Callable[[List[Tuple[Any, Any]]], float]]
        ] = None,
        test_states_number: int = 10_000,
        step_start_value: int = 0,
        episode_start_value: int = 0,
    ):
        """
        A wrapper on FOOTSIES agents which logs the specified training metrics.

        Parameters
        ----------
        agent: FootsiesAgentBase
            the agent that will be wrapped
        log_frequency: int
            how frequent, in terms of time steps, should logs related to time steps be written
        log_dir: str
            to which directory will logs be written

        episode_reward: bool
            the reward per episode. Logged independently of the log frequency
        average_reward: bool
            the exponentially weighted average of the reward per time step
        average_reward_coef: float
            the coefficient of the exponentially weighted average of the reward. If `None`, will be `1 - 1 / log_frequency`
        win_rate: bool
            the agent's win rate. Only makes sense for the FOOTSIES environment
        truncation: bool
            the number of truncated episodes
        episode_length: bool
            the average length of the episodes
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
        step_start_value: int
            the value with which the time step counter will begin.
            This is useful if one training run is stopped, and if future logs should continue after the previous training run
        episode_start_value: int
            the value with which the episode counter will begin.
            This is useful if one training run is stopped, and if future logs should continue after the previous training run
        """
        self.agent = agent
        self.log_frequency = log_frequency
        self.episode_reward_enabled = episode_reward
        self.average_reward_enabled = average_reward
        self.average_reward_coef = average_reward_coef if average_reward_coef is not None else (1 - 1 / log_frequency)
        self.win_rate_enabled = win_rate
        self.truncation_enabled = truncation
        self.episode_length_enabled = episode_length
        self.network_histograms = (
            [] if network_histograms is None else network_histograms
        )
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
        self.current_step = step_start_value
        self.current_episode = episode_start_value
        self.episode_reward = 0
        self.episode_length = 0
        self.average_reward = 0 # exponentially weighted average
        self.total_wins = 0
        self.total_terminated_episodes = 0

    def act(self, obs, *args, **kwargs) -> "any":
        return self.agent.act(obs, *args, **kwargs)

    def update(self, next_obs, reward: float, terminated: bool, truncated: bool, info: dict):
        self.agent.update(next_obs, reward, terminated, truncated, info)

        self.episode_reward += reward
        self.episode_length += 1
        self.average_reward = self.average_reward_coef * self.average_reward + (1 - self.average_reward_coef) * reward
        self.current_step += 1

        # Update the win rate tracker (only really valid for FOOTSIES)
        if terminated:
            if reward > 0.0:
                self.total_wins += 1
            self.total_terminated_episodes += 1

        # Write logs (at the episode level)
        if terminated or truncated:
            self.current_episode += 1
            
            if self.episode_reward_enabled:
                self.summary_writer.add_scalar(
                    "Performance/Episode reward",
                    self.episode_reward,
                    self.current_episode,
                )

            if self.episode_length_enabled:
                self.summary_writer.add_scalar(
                    "Training/Episode length",
                    self.episode_length,
                    self.current_episode,
                )
            
            if self.truncation_enabled:
                self.summary_writer.add_scalar(
                    "Training/Episode truncations",
                    1 if truncated else 0,
                    self.current_episode,
                )

            self.episode_reward = 0
            self.episode_length = 0

        # Write logs (at the time step level)
        if self.current_step % self.log_frequency == 0:
            if self.average_reward_enabled:
                self.summary_writer.add_scalar(
                    "Performance/Average reward",
                    self.average_reward,
                    self.current_step,
                )
            if self.win_rate_enabled:
                self.summary_writer.add_scalar(
                    "Performance/Win rate",
                    (self.total_wins / self.total_terminated_episodes) if self.total_terminated_episodes > 0 else 0.5,
                    self.current_step,
                )

            for network in self.network_histograms:
                for layer_name, layer in network.named_parameters():
                    try:
                        self.summary_writer.add_histogram(
                            layer_name, layer, self.current_step
                        )
                    except ValueError as e:
                        raise RuntimeError(f"Oops, exception happened when adding network histogram: '%s', here are the parameters: %s", e, layer) from e

            for tag, evaluator in self.custom_evaluators:
                self.summary_writer.add_scalar(tag, evaluator(), self.current_step)

            for tag, evaluator in self.custom_evaluators_over_test_states:
                self.summary_writer.add_scalar(
                    tag, evaluator(self.test_states), self.current_step
                )

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

    def load(self, folder_path: str):
        self.agent.load(folder_path)

    def save(self, folder_path: str):
        self.agent.save(folder_path)

    def extract_policy(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        return self.agent.extract_policy(env)
