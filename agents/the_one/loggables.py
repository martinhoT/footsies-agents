from .agent import FootsiesAgent
from agents.a2c.loggables import get_loggables as get_a2c_loggables


def get_loggables(agent: FootsiesAgent):
    network_histograms = []
    custom_evaluators = []

    if agent.representation is not None:
        network_histograms.append(agent.representation)
    if agent.game_model is not None:
        network_histograms.append(agent.game_model)
        custom_evaluators.append(("Learning/Loss_Game_Model", agent.evaluate_average_loss_game_model))
    if agent.opponent_model is not None:
        network_histograms.append(agent.opponent_model)
        custom_evaluators.append(("Learning/Loss_Opponent_Model", agent.evaluate_average_loss_opponent_model))

    a2c_loggables = get_a2c_loggables(agent.a2c)
    network_histograms.extend(a2c_loggables["network_histograms"])
    custom_evaluators.extend(a2c_loggables["custom_evaluators"])

    return {
        "network_histograms": network_histograms,
        "custom_evaluators": custom_evaluators,
        "custom_evaluators_over_test_states": a2c_loggables["custom_evaluators_over_test_states"],
    }
