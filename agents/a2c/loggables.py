from .agent import FootsiesAgent


def get_loggables(agent: FootsiesAgent):
    return {
        "network_histograms": [
            agent.actor,
            agent.critic,
        ],
        "custom_evaluators": [
            ("Learning/Delta", agent.evaluate_average_delta),
        ],
        "custom_evaluators_over_test_states": [],
    }
