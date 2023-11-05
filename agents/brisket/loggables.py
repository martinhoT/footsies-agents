from .agent import FootsiesAgent

def get_loggables(agent: FootsiesAgent):
    return {
        "network_histograms": [agent.q_network],
        "custom_evaluators": [("Epsilon", lambda: agent.epsilon)],
        "custom_evaluators_over_test_states": [("Average Q-value", agent.evaluate_q_network)]
    }
    