import numpy as np
from typing import Callable, TypeVar


T = TypeVar("T")


# TODO: observation history is missing!
class ReactionTimeEmulator:
    def __init__(
        self,
        include_inaction: bool,
        inaction_probability: float,
        multiplier: float,
        additive: float,
    ):
        """
        Emulator of human choice reaction time following Hick's law.

        Parameters
        ----------
        - `include_inaction`: whether to explicitly consider the action of doing nothing. If the agent action distribution already considers this, then this value should be `False`
        - `inaction_probability`: if `include_inaction` is `True`, then this is the probability associated to the action of doing nothing
        - `multiplier`: the entropy multiplier in the reaction time formula
        - `additive`: the additive constant in the reaction time formula
        """
        self.include_inaction = include_inaction
        self.inaction_probability = inaction_probability
        self.multiplier = multiplier
        self.additive = additive

    @staticmethod
    def entropy(distribution: np.ndarray) -> float:
        """Get entropy of a probability distribution, measured in bits."""
        return -np.nansum(distribution * np.log2(distribution))

    # TODO: consider only options from the opponent that can actually be done (doesn't make sense to use when the opponent can't do anything)
    # TODO: maybe hardcode the environment model, and simply slap the opponent action into the observation
    def decision_distribution(
        self,
        observation: T,
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
        
        opponent_distribution = opponent_distribution_source(observation).reshape((-1, 1))
        
        agent_distribution_matrix = np.array([
            # Get an agent action probability distribution for every possible near future
            agent_distribution_source(
                # Calculate the next observation assuming the agent did nothing, to see what will happen in the near future
                environment_model(observation, 0, opponent_action)
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

        if self.include_inaction:
            inaction_distribution = np.array([self.inaction_probability, 1.0 - self.inaction_probability])
            entropy = self.entropy(inaction_distribution) + self.inaction_probability * self.entropy(decision_distribution)
    
        else:
            entropy = self.entropy(decision_distribution)

        if previous_reaction_time is None:
            previous_reaction_time = float("+inf")

        return min(previous_reaction_time + 1, np.ceil(self.multiplier * entropy + self.additive))
