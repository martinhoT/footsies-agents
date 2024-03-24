from .agent import FootsiesAgent
from agents.a2c.loggables import get_loggables as get_a2c_loggables


def get_loggables(agent: FootsiesAgent):
    network_histograms = []
    custom_evaluators = []
    custom_evaluators_over_test_states = []

    if agent.representation is not None:
        network_histograms.append(agent.representation)
    if agent.game_model is not None:
        network_histograms.append(agent.game_model)
        custom_evaluators.append(("Learning/Loss (game model)", agent.evaluate_average_loss_game_model_and_clear))
    if agent.opponent_model is not None:
        network_histograms.append(agent.opponent_model.network)
        custom_evaluators.append(("Learning/Loss (opponent model)", agent.evaluate_average_loss_opponent_model_and_clear))
        custom_evaluators_over_test_states.append(("Learning/Entropy (opponent model)", agent.evaluate_average_opponent_model_entropy))

    a2c_loggables = get_a2c_loggables(agent.a2c)
    network_histograms.extend(a2c_loggables["network_histograms"])
    custom_evaluators.extend(a2c_loggables["custom_evaluators"])
    custom_evaluators_over_test_states.extend(a2c_loggables["custom_evaluators_over_test_states"])

    return {
        "network_histograms": network_histograms,
        "custom_evaluators": custom_evaluators,
        "custom_evaluators_over_test_states": custom_evaluators_over_test_states,
    }
