from .agent import FootsiesAgent


def get_loggables(agent: FootsiesAgent):
    return {
        "network_histograms": [agent.q_network],
        "custom_evaluators": [
            ("Training/Epsilon", lambda: agent.epsilon),
            ("Learning/Loss", agent.evaluate_average_loss_and_clear)
        ],
        "custom_evaluators_over_test_states": [
            ("Learning/Average Q-value", agent.evaluate_average_q_value),
            ("Learning/Average action entropy", agent.evaluate_average_action_entropy),
            ("Learning/Uncertainty", agent.evaluate_average_uncertainty),
        ],
    }
