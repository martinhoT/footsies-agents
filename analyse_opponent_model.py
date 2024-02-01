import dearpygui.dearpygui as dpg
import numpy as np
from gymnasium.wrappers.flatten_observation import FlattenObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.moves import FootsiesMove, footsies_move_index_to_move
from analysis import Analyser, footsies_move_from_one_hot
from agents.mimic.agent import FootsiesAgent as OpponentModelAgent
from agents.utils import FOOTSIES_ACTION_MOVES
from main import load_agent_model


def load_predicted_battle_state(analyser: Analyser):
    analyser.p1_guard = dpg.get_value("p1_guard_previous")
    analyser.p2_guard = dpg.get_value("p2_guard_previous")
    analyser.p1_position = dpg.get_value("p1_position_previous")
    analyser.p2_position = dpg.get_value("p2_position_previous")
    analyser.p1_move = FootsiesMove[dpg.get_value("p1_move_previous")]
    analyser.p2_move = FootsiesMove[dpg.get_value("p2_move_previous")]
    analyser.p1_move_progress = dpg.get_value("p1_move_progress_previous")
    analyser.p2_move_progress = dpg.get_value("p2_move_progress_previous")
    
    analyser.load_battle_state(analyser.custom_battle_state, require_update=False)


def include_mimic_dpg_elements(analyser: Analyser):
    dpg.add_text("Prediced current state based on previous state")
    # Previous state
    with dpg.table():
        dpg.add_table_column(label="Property")
        dpg.add_table_column(label="P1")
        dpg.add_table_column(label="P2")

        with dpg.table_row():
            dpg.add_text("Guard")
            dpg.add_slider_int(min_value=0, max_value=3, tag="p1_guard_previous", enabled=False)
            dpg.add_slider_int(min_value=0, max_value=3, tag="p2_guard_previous", enabled=False)
        
        with dpg.table_row():
            dpg.add_text("Position")
            dpg.add_slider_float(min_value=-4.4, max_value=4.4, tag="p1_position_previous", enabled=False)
            dpg.add_slider_float(min_value=-4.4, max_value=4.4, tag="p2_position_previous", enabled=False)

        with dpg.table_row():
            dpg.add_text("Move")
            dpg.add_combo([m.name for m in footsies_move_index_to_move], tag="p1_move_previous", enabled=False)
            dpg.add_combo([m.name for m in footsies_move_index_to_move], tag="p2_move_previous", enabled=False)

        with dpg.table_row():
            dpg.add_text("Move progress")
            dpg.add_slider_float(min_value=0, max_value=1, tag="p1_move_progress_previous", enabled=False)
            dpg.add_slider_float(min_value=0, max_value=1, tag="p2_move_progress_previous", enabled=False)
    
    with dpg.group(horizontal=True):
        dpg.add_text("Opponent move actually performed")
        dpg.add_combo([m.name for m in FOOTSIES_ACTION_MOVES], tag="agent_action", callback=update_prediction)
    
    with dpg.group(horizontal=True):
        dpg.add_button(label="Apply", callback=lambda: load_predicted_battle_state(analyser=analyser))


def update_prediction(observation, agent_action, opponent_action):
    next_obs = agent.predict(observation, agent_action, opponent_action).squeeze(0)

    # TODO: how should I handle cases where the model doesn't predict any move?
    p1_move = footsies_move_from_one_hot(next_obs[2:17]) if np.any(next_obs[2:17] > 0.0) else FootsiesMove.DEAD
    p2_move = footsies_move_from_one_hot(next_obs[17:32]) if np.any(next_obs[17:32] > 0.0) else FootsiesMove.DEAD

    dpg.set_value("p1_guard_previous", round(next_obs[0] * 3))
    dpg.set_value("p2_guard_previous", round(next_obs[1] * 3))
    dpg.set_value("p1_move_previous", p1_move.name)
    dpg.set_value("p2_move_previous", p2_move.name)
    dpg.set_value("p1_move_progress_previous", next_obs[32])
    dpg.set_value("p2_move_progress_previous", next_obs[33])
    dpg.set_value("p1_position_previous", next_obs[34] * 4.4)
    dpg.set_value("p2_position_previous", next_obs[35] * 4.4)
    dpg.set_value("agent_action", footsies_move_index_to_move[agent_action].name)
    dpg.set_value("opponent_action", footsies_move_index_to_move[opponent_action].name)


def predict_next_action(analyser: Analyser):
    if analyser.previous_observation is None:
        return

    agent: OpponentModelAgent = analyser.agent

    # Note: we need to consider the current information when determining the players' moves, but the previous observation!
    observation = analyser.previous_observation
    agent_action = agent.simplify_action(analyser.current_info["p1_move"])
    opponent_action = agent.simplify_action(analyser.current_info["p2_move"])

    update_prediction(observation, agent_action, opponent_action)


if __name__ == "__main__":

    footsies_env = FootsiesEnv(
        game_path="../Footsies-Gym/Build/FOOTSIES.x86_64",
        game_port=15000,
        opponent_port=15001,
        render_mode="human",
        sync_mode="synced_non_blocking",
        fast_forward=False,
    )

    env = FootsiesActionCombinationsDiscretized(
        FlattenObservation(
            FootsiesNormalized(footsies_env)
        )
    )

    agent = OpponentModelAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        opponent_action_dim=10,
        by_primitive_actions=False,
    )

    load_agent_model(agent, "mimic")

    analyser = Analyser(
        env=env,
        agent=agent,
        custom_elements_callback=include_mimic_dpg_elements,
        custom_state_update_callback=predict_next_action,
    )
    analyser.start()
