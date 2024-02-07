from .agent import FootsiesAgent


def get_loggables(agent: FootsiesAgent):
    return {
        "network_histograms": [
            agent.game_model,
        ],
        "custom_evaluators": [
            ("Learning/Loss_Guard", agent.evaluate_average_loss_guard),
            ("Learning/Loss_Move_P1", agent.evaluate_average_loss_move_p1),
            ("Learning/Loss_Move_P2", agent.evaluate_average_loss_move_p2),
            ("Learning/Loss_MoveProgress", agent.evaluate_average_loss_move_progress),
            ("Learning/Loss_Position", agent.evaluate_average_loss_position),
            ("Learning/Loss", agent.evaluate_average_loss), # the evaluate loss one should be at the end since it's the one that clears the denominator counter
        ],
        "custom_evaluators_over_test_states": [],
    }
