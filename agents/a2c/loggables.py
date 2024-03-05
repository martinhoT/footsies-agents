from .agent import FootsiesAgent
from agents.ql.ql import QTable
from agents.a2c.a2c import CriticNetwork


def get_loggables(agent: FootsiesAgent):
    network_histograms = [
        agent.actor
    ]
    custom_evaluators = [
        ("Learning/Delta", agent.evaluate_average_delta),
    ]
    custom_evaluators_over_test_states = [
        ("Learning/Policy entropy", agent.evaluate_average_policy_entropy),
    ]
    
    if isinstance(agent.critic, QTable):
        custom_evaluators.extend([
            ("Learning/Q-table sparsity", agent.critic.sparsity),
            ("Learning/Q-table size", lambda: len(agent.critic.table)),
            ("Learning/Q-table error", agent.evaluate_average_qtable_error),
        ])

    if isinstance(agent.critic, CriticNetwork):
        network_histograms.append(agent.critic)

    return {
        "network_histograms": network_histograms,
        "custom_evaluators": custom_evaluators,
        "custom_evaluators_over_test_states": custom_evaluators_over_test_states,
    }
