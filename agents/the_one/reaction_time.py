import numpy as np
from typing import Callable, TypeVar
from collections import deque
from itertools import islice


T = TypeVar("T")


class ReactionTimeEmulator:
    def __init__(
        self,
        inaction_probability: float = 0.1,
        multiplier: float = 1,
        additive: float = 0,
        history_size: int = 30,
    ):
        """
        Emulator of human choice reaction time following Hick's law.

        Parameters
        ----------
        - `inaction_probability`: this is the probability associated to the action of doing nothing. If the agent action distribution already considers this, then this value should be 0
        - `multiplier`: the entropy multiplier in the reaction time formula
        - `additive`: the additive constant in the reaction time formula
        - `history_size`: the size of the observation history
        """
        self._inaction_probability = inaction_probability
        self._multiplier = multiplier
        self._additive = additive
        self._history_size = history_size

        self._observations = deque([], maxlen=history_size)
        self._previous_reaction_time = None
        self._constant = False

        # Pre-made values that are useful for computation
        inaction_distribution = self.bernoulli_distribution(self._inaction_probability)
        self._inaction_entropy = self.entropy(inaction_distribution)

    @staticmethod
    def entropy(distribution: np.ndarray, safety: float = 1e-8) -> float:
        """Get entropy of a probability distribution, measured in nats."""
        # Should the safety value be included in the probability portion of the formula? That makes the sum of probabilities not be 1. Probably not.
        # We measure entropy in nats to match the way entropy is measured in PyTorch by default.
        return -np.nansum((distribution) * np.log2(distribution + safety))
    
    @staticmethod
    def bernoulli_distribution(probability: float) -> np.ndarray:
        return np.array([probability, 1.0 - probability])

    @property
    def inaction_probability(self) -> float:
        """The probability of doing nothing."""
        return self._inaction_probability

    @inaction_probability.setter
    def inaction_probability(self, value: float):
        self._inaction_probability = value

        inaction_distribution = self.bernoulli_distribution(self._inaction_probability)
        self._inaction_entropy = self.entropy(inaction_distribution)

    def confine_to_range(self, minimum: int, maximum: int, agent_n_actions: int):
        """
        Define the multiplier and additive parameters to confine the reaction time to a defined range.
        Make sure to apply this operation after other parameters of the emulator, such as the inaction probability, are set.
        """
        if maximum >= self._history_size:
            raise ValueError(f"the maximum value cannot be equal to or greater than the defined history size ({maximum} >= {self._history_size})")
        
        maximum_entropy = self.maximum_decision_entropy(agent_n_actions)
        b = (maximum - minimum) / (maximum_entropy - self._inaction_entropy)
        c = minimum - b * self._inaction_entropy

        self._multiplier = b
        self._additive = c

    def maximum_decision_entropy(self, agent_n_actions: int) -> float:
        """Calculate the maximum possible decision entropy."""
        return self._inaction_entropy + (1 - self.inaction_probability) * np.log2(agent_n_actions)

    def reaction_time(self, decision_entropy: float, previous_reaction_time: int | None = None) -> int:
        """
        Calculate reaction time in time steps given a decision distribution.
        
        Parameters
        ----------
        - `decision_distribution`: the entropy of the probability distribution of the decision
        - `previous_reaction_time`: the reaction time at the previous observation, in order to bound the returned reaction time into a reasonable range.
        The returned reaction time will not be bounded if this value is `None`
        """

        decision_entropy = self._inaction_entropy + (1 - self._inaction_probability) * decision_entropy

        if previous_reaction_time is None:
            previous_reaction_time = self._history_size - 1

        computed_reaction_time = int(np.ceil(self._multiplier * decision_entropy + self._additive))
        # We need to be reasonable, reaction time should not be larger than the previous one (we can't start seeing things in the past)
        return min(previous_reaction_time + 1, computed_reaction_time)
    
    def register_observation(self, observation: T):
        """Register an observation of the environment into the observation history. Should be performed at every environment step, before perceiving an observation."""
        self._observations.append(observation)

    def perceive(self, reaction_time: int, previous_reaction_time: int | None = None) -> tuple[T, list[T]]:
        """Get an observation according to the provided reaction time, and all the observations that were skipped according to the previous reaction time. For a reaction time of 0, return the observation at the current instant."""
        if previous_reaction_time is None:
            previous_reaction_time = self._history_size - 1

        perceived_observation = self._observations[-1 - reaction_time]
        skipped_observations = list(islice(self._observations, (self._observations.maxlen - 1) - previous_reaction_time, (self._observations.maxlen - 1) - reaction_time))

        return perceived_observation, skipped_observations
    
    def register_and_perceive(
        self,
        observation: T,
        decision_entropy: float,
    ) -> tuple[T, int, list[T]]:
        """
        Perform registration of the current observation and perception of a delayed observation all in a single method, for convenience. This is the primary method that should be used at every environment step.
        The decision entropy should be provided in nats.
        
        Note: the decision entropy is calculated according to the last reaction time computed through this method, which is initially the maximum possible (perceives the oldest observation).

        Returns
        -------
        - `perceived_observation`: the observation that was perceived, `reaction_time` time steps late
        - `reaction_time`: the computed reaction time
        - `skipped_observations`: list of all observations that were skipped to get to `perceived_observation`
        """
        # Transform the decision entropy to bits
        decision_entropy = decision_entropy / np.log(2)

        if self._constant:
            reaction_time = self._additive
        else:
            reaction_time = self.reaction_time(decision_entropy, self._previous_reaction_time)

        self.register_observation(observation)
        perceived_observation, skipped_observations = self.perceive(reaction_time, self._previous_reaction_time)

        self._previous_reaction_time = reaction_time

        return perceived_observation, reaction_time, skipped_observations

    def fill_history(self, observation: T):
        """Fill the observation history with a single observation. Should be done initially, immediately after an environment reset."""
        self._observations.clear()
        self._observations.extend([observation] * self._history_size)
    
    @property
    def constant(self) -> bool:
        """Whether reaction time is a constant and thus not dependent on the decision distribution's entropy."""
        return self._constant

    @constant.setter
    def constant(self, value: bool):
        self._constant = value

    @property
    def previous_reaction_time(self) -> int:
        """The most recently perceived reaction time (last call to `register_and_perceive`)."""
        return self._previous_reaction_time

    @property
    def history_size(self) -> int:
        """The (maximum) size of the observation history."""
        return self._observations.maxlen