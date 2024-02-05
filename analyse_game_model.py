import dearpygui.dearpygui as dpg
from gymnasium.wrappers.flatten_observation import FlattenObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.moves import FootsiesMove, footsies_move_index_to_move
from analysis import Analyser, footsies_move_from_one_hot
from agents.game_model.agent import FootsiesAgent as GameModelAgent
from agents.utils import FOOTSIES_ACTION_MOVES, FOOTSIES_ACTION_MOVE_INDICES_MAP, FOOTSIES_ACTION_MOVE_INDEX_MAP
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


def include_game_model_dpg_elements(analyser: Analyser):
    dpg.add_text("Prediced current state based on the previous observation")
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
        dpg.add_text("Agent action performed on previous state")
        dpg.add_combo([m.name for m in FOOTSIES_ACTION_MOVES], tag="agent_action", callback=lambda: predict_next_state(analyser))
    
    with dpg.group(horizontal=True):
        dpg.add_text("Opponent action performed on previous state")
        dpg.add_combo([m.name for m in FOOTSIES_ACTION_MOVES], tag="opponent_action", callback=lambda: predict_next_state(analyser))

    with dpg.group(horizontal=True):
        dpg.add_button(label="Apply", callback=lambda: load_predicted_battle_state(analyser=analyser))


def update_prediction(agent: GameModelAgent, observation, agent_action, opponent_action):
    next_obs = agent.predict(observation, agent_action, opponent_action).squeeze(0)

    p1_move = footsies_move_from_one_hot(next_obs[2:17])
    p2_move = footsies_move_from_one_hot(next_obs[17:32])

    dpg.set_value("p1_guard_predicted", round(next_obs[0] * 3))
    dpg.set_value("p2_guard_predicted", round(next_obs[1] * 3))
    dpg.set_value("p1_move_predicted", p1_move.name)
    dpg.set_value("p2_move_predicted", p2_move.name)
    dpg.set_value("p1_move_progress_predicted", next_obs[32])
    dpg.set_value("p2_move_progress_predicted", next_obs[33])
    dpg.set_value("p1_position_predicted", next_obs[34] * 4.4)
    dpg.set_value("p2_position_predicted", next_obs[35] * 4.4)
    dpg.set_value("agent_action", FOOTSIES_ACTION_MOVES[agent_action].name)
    dpg.set_value("opponent_action", FOOTSIES_ACTION_MOVES[opponent_action].name)


def predict_next_state(analyser: Analyser, agent_action: int = None, opponent_action: int = None):
    if analyser.previous_observation is None:
        return

    agent: GameModelAgent = analyser.agent

    # Note: we need to consider the current information when determining the players' moves, but the previous observation!
    observation = analyser.previous_observation
    agent_action = FOOTSIES_ACTION_MOVE_INDEX_MAP[FootsiesMove[dpg.get_value("agent_action")]] if agent_action is None else agent_action
    opponent_action = FOOTSIES_ACTION_MOVE_INDEX_MAP[FootsiesMove[dpg.get_value("opponent_action")]] if opponent_action is None else opponent_action
    
    update_prediction(agent, observation, agent_action, opponent_action)


def update_info_and_predict_next_state(analyser: Analyser):
    agent_action = FOOTSIES_ACTION_MOVE_INDICES_MAP[analyser.current_info["p1_move"]]
    opponent_action = FOOTSIES_ACTION_MOVE_INDICES_MAP[analyser.current_info["p2_move"]]

    predict_next_state(analyser, agent_action, opponent_action)


if __name__ == "__main__":

    footsies_env = FootsiesEnv(
        game_path="../Footsies-Gym/Build/FOOTSIES.x86_64",
        game_port=15000,
        opponent_port=15001,
        remote_control_port=15002,
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
        by_primitive_actions=False,
        hidden_layer_sizes_specification="128",
        hidden_layer_activation_specification="LeakyReLU",
    )

    load_agent_model(agent, "game_model_one_layer_slight_scale")

    analyser = Analyser(
        env=env,
        agent=agent,
        custom_elements_callback=include_game_model_dpg_elements,
        custom_state_update_callback=update_info_and_predict_next_state,
    )
    analyser.start()
