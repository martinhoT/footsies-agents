from .agent import FootsiesAgent


def get_loggables(agent: FootsiesAgent):
    return {
        "network_histograms": [
            agent.game_model,
        ],
        "custom_evaluators": [
            ("Learning/Loss", agent.evaluate_average_loss),
        ],
        "custom_evaluators_over_test_states": [],
    }
