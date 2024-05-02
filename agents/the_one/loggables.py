from .agent import TheOneAgent
from agents.a2c.loggables import get_loggables as get_a2c_loggables
from agents.mimic.loggables import get_loggables as get_mimic_loggables
from agents.game_model.loggables import get_loggables as get_game_model_loggables


def get_loggables(agent: TheOneAgent):
    network_histograms = []
    custom_evaluators = []
    custom_evaluators_over_test_states = []

    a2c_loggables = get_a2c_loggables(agent.a2c)
    network_histograms.extend(a2c_loggables["network_histograms"])
    custom_evaluators.extend(a2c_loggables["custom_evaluators"])
    custom_evaluators_over_test_states.extend(a2c_loggables["custom_evaluators_over_test_states"])

    if agent.opp is not None:
        mimic_loggables = get_mimic_loggables(agent.opp)
        network_histograms.extend(mimic_loggables["network_histograms"])
        custom_evaluators.extend(mimic_loggables["custom_evaluators"])
        custom_evaluators_over_test_states.extend(mimic_loggables["custom_evaluators_over_test_states"])

    if agent.gm is not None:
        game_model_loggables = get_game_model_loggables(agent.gm)
        network_histograms.extend(game_model_loggables["network_histograms"])
        custom_evaluators.extend(game_model_loggables["custom_evaluators"])
        custom_evaluators_over_test_states.extend(game_model_loggables["custom_evaluators_over_test_states"])

    if agent.reaction_time_emulator is not None:
        custom_evaluators.append(("Training/Reaction time", agent.evaluate_average_reaction_time))

    custom_evaluators.append(("Training/Act elapsed time (ns)", agent.evaluate_act_elapsed_time_sn))

    return {
        "network_histograms": network_histograms,
        "custom_evaluators": custom_evaluators,
        "custom_evaluators_over_test_states": custom_evaluators_over_test_states,
    }
