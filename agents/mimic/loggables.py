from .agent import FootsiesAgent


def get_loggables(agent: FootsiesAgent):
    network_histograms = []
    custom_evaluators = []
    custom_evaluators_over_test_states = []
    
    if agent.p1_model is not None:
        network_histograms.append(agent.p1_model.network)
        custom_evaluators.append(("Learning/Loss of P1's model", agent.evaluate_p1_average_loss_and_clear))
        custom_evaluators.append(("Learning/Scar size of P1's model", lambda: agent.p1_model.number_of_scars))
        custom_evaluators_over_test_states.append(("Learning/Entropy of P1's model", lambda test_states: agent.evaluate_decision_entropy(test_states, p1=True)))

    if agent.p2_model is not None:
        network_histograms.append(agent.p2_model.network)
        custom_evaluators.append(("Learning/Loss of P2's model", agent.evaluate_p2_average_loss_and_clear))
        custom_evaluators.append(("Learning/Scar size of P2's model", lambda: agent.p2_model.number_of_scars))
        custom_evaluators_over_test_states.append(("Learning/Entropy of P2's model", lambda test_states: agent.evaluate_decision_entropy(test_states, p1=False)))
    
    if agent.p1_model is not None and agent.p2_model is not None:
        custom_evaluators_over_test_states.append(("Learning/KL-Divergence between player models (P1 to P2)", agent.evaluate_divergence_between_players))

    return {
        "network_histograms": network_histograms,
        "custom_evaluators": custom_evaluators,
        "custom_evaluators_over_test_states": custom_evaluators_over_test_states,
    }
