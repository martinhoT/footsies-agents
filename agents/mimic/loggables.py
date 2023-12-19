from .agent import FootsiesAgent


def get_loggables(agent: FootsiesAgent):
    return {
        "network_histograms": [
            agent.p1_model.network,
            agent.p2_model.network,
        ],
        "custom_evaluators": [
            ("Learning/Loss_P1", lambda: agent.evaluate_average_loss_and_clear(True)),
            ("Learning/Loss_P2", lambda: agent.evaluate_average_loss_and_clear(False)),
            ("Learning/Divergence", agent.evaluate_divergence_between_players),
        ],
        "custom_evaluators_over_test_states": [
            ("Learning/Average Q-value", agent.evaluate_average_q_value),
            ("Learning/Average action entropy", agent.evaluate_average_action_entropy),
            ("Learning/Uncertainty", agent.evaluate_average_uncertainty),
        ],
    }
