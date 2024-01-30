import copy
import dearpygui.dearpygui as dpg
import numpy as np
from datetime import datetime
from typing import Callable
from gymnasium import Env
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.spaces.utils import flatten
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.state import FootsiesBattleState
from footsies_gym.moves import FootsiesMove, footsies_move_index_to_move, footsies_move_id_to_index
from agents.base import FootsiesAgentBase
from analysis import Analyser, footsies_move_from_one_hot
from agents.game_model.agent import FootsiesAgent as GameModelAgent
from main import load_agent_model


def load_predicted_battle_state(analyser: Analyser):
    analyser.p1_guard = dpg.get_value("p1_guard_predicted")
    analyser.p2_guard = dpg.get_value("p2_guard_predicted")
    analyser.p1_position = dpg.get_value("p1_position_predicted")
    analyser.p2_position = dpg.get_value("p2_position_predicted")
    analyser.p1_move = FootsiesMove[dpg.get_value("p1_move_predicted")]
    analyser.p2_move = FootsiesMove[dpg.get_value("p2_move_predicted")]
    analyser.p1_move_progress = dpg.get_value("p1_move_progress_predicted")
    analyser.p2_move_progress = dpg.get_value("p2_move_progress_predicted")
    
    analyser.load_battle_state(analyser.custom_battle_state, require_update=False)


def load_predicted_battle_state_and_advance(analyser: Analyser):
    load_predicted_battle_state(analyser=analyser)
    analyser.advance()


def include_game_model_dpg_elements(analyser: Analyser):
    dpg.add_text("Prediced next state")
    # Predicted state
    with dpg.table():
        dpg.add_table_column(label="Property")
        dpg.add_table_column(label="P1")
        dpg.add_table_column(label="P2")

        with dpg.table_row():
            dpg.add_text("Guard")
            dpg.add_slider_int(min_value=0, max_value=3, tag="p1_guard_predicted", enabled=False)
            dpg.add_slider_int(min_value=0, max_value=3, tag="p2_guard_predicted", enabled=False)
        
        with dpg.table_row():
            dpg.add_text("Position")
            dpg.add_slider_float(min_value=-4.4, max_value=4.4, tag="p1_position_predicted", enabled=False)
            dpg.add_slider_float(min_value=-4.4, max_value=4.4, tag="p2_position_predicted", enabled=False)

        with dpg.table_row():
            dpg.add_text("Move")
            dpg.add_combo([m.name for m in footsies_move_index_to_move], tag="p1_move_predicted", enabled=False)
            dpg.add_combo([m.name for m in footsies_move_index_to_move], tag="p2_move_predicted", enabled=False)

        with dpg.table_row():
            dpg.add_text("Move progress")
            dpg.add_slider_float(min_value=0, max_value=1, tag="p1_move_progress_predicted", enabled=False)
            dpg.add_slider_float(min_value=0, max_value=1, tag="p2_move_progress_predicted", enabled=False)
    
    with dpg.group(horizontal=True):
        dpg.add_button(label="Apply", callback=lambda: load_predicted_battle_state(analyser=analyser))
        dpg.add_button(label="Apply and Advance", callback=lambda: load_predicted_battle_state_and_advance(analyser=analyser))


def prediction_on_state_update(analyser: Analyser):
    agent: GameModelAgent = analyser.agent

    obs = analyser.current_observation
    agent_action = agent.simplify_action(
        footsies_move_id_to_index[
            footsies_move_from_one_hot(obs[2:17]).value.id
        ]
    )
    opponent_action = agent.simplify_action(
        footsies_move_id_to_index[
            footsies_move_from_one_hot(obs[17:32]).value.id
        ]
    )

    next_obs = agent.predict(obs, agent_action, opponent_action).squeeze(0)

    # TODO: how should I handle cases where the model doesn't predict any move?
    p1_move = footsies_move_from_one_hot(next_obs[2:17]) if np.any(next_obs[2:17] > 0.0) else FootsiesMove.DEAD
    p2_move = footsies_move_from_one_hot(next_obs[17:32]) if np.any(next_obs[17:32] > 0.0) else FootsiesMove.DEAD

    dpg.set_value("p1_guard_predicted", round(next_obs[0] * 3))
    dpg.set_value("p2_guard_predicted", round(next_obs[1] * 3))
    dpg.set_value("p1_move_predicted", p1_move.name)
    dpg.set_value("p2_move_predicted", p2_move.name)
    dpg.set_value("p1_move_progress_predicted", next_obs[32])
    dpg.set_value("p2_move_progress_predicted", next_obs[33])
    dpg.set_value("p1_position_predicted", next_obs[34] * 4.4)
    dpg.set_value("p2_position_predicted", next_obs[35] * 4.4)


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

    agent = GameModelAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        opponent_action_dim=10,
        by_primitive_actions=False,
    )

    load_agent_model(agent, "game_model")

    analyser = Analyser(
        env=env,
        agent=agent,
        custom_elements_callback=include_game_model_dpg_elements,
        custom_state_update_callback=prediction_on_state_update,
    )
    analyser.start()
