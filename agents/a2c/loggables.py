from .agent import A2CAgent
from agents.ql.ql import QFunctionTable, QFunctionNetwork
from agents.a2c.a2c import ValueNetwork, A2CQLearner
from functools import partial
from torch import nn
from agents.logger import CustomEvaluator, CustomEvaluatorOverTestStates


def q_network_target_update_step(*, qf_network: QFunctionNetwork) -> int:
    return qf_network._current_update_step


def get_loggables(agent: A2CAgent) -> dict[str, list]:
    network_histograms: list[nn.Module] = [
        agent._actor
    ]
    custom_evaluators: list[CustomEvaluator] = [
        ("Learning/Delta", agent.evaluate_average_delta),
    ]
    custom_evaluators_over_test_states: list[CustomEvaluatorOverTestStates] = [
        ("Learning/Policy entropy", agent.evaluate_average_policy_entropy),
    ]
    
    if isinstance(agent.learner.critic, QFunctionTable):
        
        def q_table_size(critic: QFunctionTable):
            return len(critic.table)
        
        custom_evaluators.extend([
            ("Learning/Q-table sparsity", agent.learner.critic.sparsity),
            ("Learning/Q-table size", partial(q_table_size, critic=agent.learner.critic)),
        ])

    if isinstance(agent.learner, A2CQLearner):
        custom_evaluators.append(("Learning/Q-learner error", agent.evaluate_average_qtable_error))

    if isinstance(agent.learner.critic, QFunctionNetwork):
        network_histograms.append(agent.learner.critic.q_network)
        custom_evaluators.append(("Learning/Q-network target update step", partial(q_network_target_update_step, qf_network=agent.learner.critic)))

    if isinstance(agent._critic, ValueNetwork):
        network_histograms.append(agent._critic)

    return {
        "network_histograms": network_histograms,
        "custom_evaluators": custom_evaluators,
        "custom_evaluators_over_test_states": custom_evaluators_over_test_states,
    }
