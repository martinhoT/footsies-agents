from .agent import FootsiesAgent
from agents.ql.ql import QFunctionTable, QFunctionNetwork
from agents.a2c.a2c import ValueNetwork, A2CQLearner


def get_loggables(agent: FootsiesAgent) -> dict[str, list]:
    network_histograms = [
        agent._actor
    ]
    custom_evaluators = [
        ("Learning/Delta", agent.evaluate_average_delta),
    ]
    custom_evaluators_over_test_states = [
        ("Learning/Policy entropy", agent.evaluate_average_policy_entropy),
    ]
    
    if isinstance(agent._critic, QFunctionTable):
        custom_evaluators.extend([
            ("Learning/Q-table sparsity", agent._critic.sparsity),
            ("Learning/Q-table size", lambda: len(agent._critic.table)),
        ])

    if isinstance(agent.learner, A2CQLearner):
        custom_evaluators.append(("Learning/Q-learner error", agent.evaluate_average_qtable_error))

    if isinstance(agent._critic, QFunctionNetwork):
        network_histograms.append(agent._critic.q_network)

    if isinstance(agent._critic, ValueNetwork):
        network_histograms.append(agent._critic)

    return {
        "network_histograms": network_histograms,
        "custom_evaluators": custom_evaluators,
        "custom_evaluators_over_test_states": custom_evaluators_over_test_states,
    }
