from .agent import FootsiesAgent


def get_loggables(agent: FootsiesAgent):
    return {
        "network_histograms": [],
        "custom_evaluators": [
            ("Learning/Loss_P1", lambda: agent.evaluate_average_loss_and_clear(True)),
            ("Learning/Loss_P2", lambda: agent.evaluate_average_loss_and_clear(False)),
            ("Learning/Performance_P1", lambda: agent.evaluate_performance(True)),
            ("Learning/Performance_P2", lambda: agent.evaluate_performance(False)),
            ("Learning/Divergence", agent.evaluate_divergence_between_players),
        ],
        "custom_evaluators_over_test_states": [
            ("Learning/Uncertainty_P1", lambda test_states: agent.evaluate_decision_entropy(test_states, True)),
            ("Learning/Uncertainty_P2", lambda test_states: agent.evaluate_decision_entropy(test_states, False)),
        ],
    }
