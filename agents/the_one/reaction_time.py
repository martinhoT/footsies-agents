import numpy as np
from typing import Callable, TypeVar
from collections import deque


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
        self._previous_reaction_time = float("+inf")

        # Pre-made values that are useful for computation
        inaction_distribution = self.bernoulli_distribution(self._inaction_probability)
        self._inaction_entropy = self.entropy(inaction_distribution)

    @staticmethod
    def entropy(distribution: np.ndarray) -> float:
        """Get entropy of a probability distribution, measured in bits"""
        return -np.nansum(distribution * np.log2(distribution))
    
    @staticmethod
    def bernoulli_distribution(probability: float) -> np.ndarray:
        return np.array([probability, 1.0 - probability])

    @property
    def inaction_probability(self) -> float:
        """The probability of doing nothing"""
        return self._inaction_probability

    @inaction_probability.setter
    def inaction_probability(self, value: float):
        self._inaction_probability = value

        inaction_distribution = self.bernoulli_distribution(self._inaction_probability)
        self._inaction_entropy = self.entropy(inaction_distribution)

    def confine_to_range(self, minimum: float, maximum: float, agent_n_actions: int):
        """
        Define the multiplier and additive parameters to confine the reaction time to a defined range.
        Make sure to apply this operation after other parameters of the emulator, such as the inaction probability, are set
        """
        if maximum >= self._history_size:
            raise ValueError(f"the maximum value cannot be equal to or greater than the defined history size ({maximum} >= {self._history_size})")
        
        maximum_entropy = self.maximum_decision_entropy(agent_n_actions)
        b = (maximum - minimum) / (maximum_entropy - self._inaction_entropy)
        c = minimum - b * self._inaction_entropy

        self._multiplier = b
        self._additive = c

    def maximum_decision_entropy(self, agent_n_actions: int) -> float:
        """Calculate the maximum possible decision entropy"""
        return self._inaction_entropy + (1 - self.inaction_probability) * np.log2(agent_n_actions)

    # TODO: consider only options from the opponent that can actually be done (doesn't make sense to use when the opponent can't do anything)
    # TODO: maybe hardcode the environment model, and simply slap the opponent action into the observation
    def decision_distribution(
        self,
        perceived_observation: T,
        agent_distribution_source: Callable[[T], np.ndarray],
        opponent_distribution_source: Callable[[T], np.ndarray],
        environment_model: Callable[[T, int, int], T],
    ) -> np.ndarray:
        """
        Calculate the probability distribution of the decision as a function of the agent and opponent action distributions on the previous state.
        It's assumed that the opponent actions can be imposed directly on the observation.
        
        Parameters
        ----------
        - `observation`: the observation on which the decision distribution is computed
        - `agent_distribution_source`: function that, given an observation, provides the agent's action probability distribution
        - `opponent_distribution_source`: function that, given an observation, provides the opponent's action probability distribution
        - `environment_model`: a model of the environment, that provides the next observation given the current observation, agent action and opponent action.
        It is assumed that the action `0` signifies inaction (doing nothing)
        """
        
        opponent_distribution = opponent_distribution_source(perceived_observation).reshape((-1, 1))
        
        agent_distribution_matrix = np.array([
            # Get an agent action probability distribution for every possible near future
            agent_distribution_source(
                # Calculate the next observation assuming the agent did nothing, to see what will happen in the near future
                environment_model(perceived_observation, 0, opponent_action)
            )
            for opponent_action in range(len(opponent_distribution))
        ])
        
        return np.sum(opponent_distribution * agent_distribution_matrix, axis=0).squeeze()

    def reaction_time(self, decision_distribution: np.ndarray, previous_reaction_time: int = None) -> int:
        """
        Calculate reaction time in time steps given a decision distribution.
        
        Parameters
        ----------
        - `decision_distribution`: the probability distribution of the decision, ideally calculated using the `decision_distribution()` method
        - `previous_reaction_time`: the reaction time at the previous observation, in order to bound the returned reaction time into a reasonable range.
        The returned reaction time will not be bounded if this value is `None`
        """

        decision_entropy = self._inaction_entropy + (1 - self._inaction_probability) * self.entropy(decision_distribution)

        if previous_reaction_time is None:
            previous_reaction_time = float("+inf")

        return min(previous_reaction_time + 1, np.ceil(self._multiplier * decision_entropy + self._additive))
    
    def register_observation(self, observation: T):
        """Register an observation of the environment into the observation history. Should be performed at every environment step, before perceiving an observation"""
        self._observations.append(observation)

    def perceive(self, reaction_time: int) -> T:
        """Get an observation according to the provided reaction time. For a reaction time of 0, return the observation at the current instant"""
        return self._observations[-(reaction_time + 1)]
    
    def register_and_perceive(
        self,
        observation: T,
        agent_distribution_source: Callable[[T], np.ndarray],
        opponent_distribution_source: Callable[[T], np.ndarray],
        environment_model: Callable[[T, int, int], T],
    ) -> tuple[T, int]:
        """
        Perform registration and perception of a delayed observation all in a single method, for convenience. This is the primary method that should be used at every environment step
        
        Note: the decision entropy is calculated according to the last reaction time computed through this method, which is initially positive infinity (perceives the oldest observation)
        """
        
        last_perceived_observation = self.perceive(
            min(self._previous_reaction_time, self._history_size - 1)
        )

        decision_distribution = self.decision_distribution(
            last_perceived_observation,
            agent_distribution_source,
            opponent_distribution_source,
            environment_model
        )

        reaction_time = self.reaction_time(decision_distribution, self._previous_reaction_time)

        self.register_observation(observation)
        perceived_observation = self.perceive(reaction_time)

        self._previous_reaction_time = reaction_time

        return perceived_observation, reaction_time

    def fill_history(self, observation: T):
        """Fill the observation history with a single observation. Should be done initially, immediately after an environment reset"""
        self._observations.clear()
        self._observations.extend([observation] * self._history_size)