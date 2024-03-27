from .agent import FootsiesAgent
from agents.a2c.loggables import get_loggables as get_a2c_loggables
from agents.mimic.loggables import get_loggables as get_mimic_loggables


def get_loggables(agent: FootsiesAgent):
    network_histograms = []
    custom_evaluators = []
    custom_evaluators_over_test_states = []

    if agent.representation is not None:
        network_histograms.append(agent.representation)
    if agent.game_model is not None:
        network_histograms.append(agent.game_model)

    a2c_loggables = get_a2c_loggables(agent.a2c)
    network_histograms.extend(a2c_loggables["network_histograms"])
    custom_evaluators.extend(a2c_loggables["custom_evaluators"])
    custom_evaluators_over_test_states.extend(a2c_loggables["custom_evaluators_over_test_states"])

    if agent.opponent_model is not None:
        mimic_loggables = get_mimic_loggables(agent.opponent_model)
        network_histograms.extend(mimic_loggables["network_histograms"])
        custom_evaluators.extend(mimic_loggables["custom_evaluators"])
        custom_evaluators_over_test_states.extend(mimic_loggables["custom_evaluators_over_test_states"])

    if agent.reaction_time_emulator is not None:
        custom_evaluators.append(("Training/Reaction time", lambda: agent.reaction_time_emulator.previous_reaction_time))

    return {
        "network_histograms": network_histograms,
        "custom_evaluators": custom_evaluators,
        "custom_evaluators_over_test_states": custom_evaluators_over_test_states,
    }
