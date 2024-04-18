import torch
import dearpygui.dearpygui as dpg
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.moves import FootsiesMove, FOOTSIES_MOVE_INDEX_TO_MOVE
from agents.game_model.game_model import GameModel, GameModelNetwork
from scripts.analysis.analysis import Analyser
from agents.game_model.agent import GameModelAgent
from agents.action import ActionMap
from scripts.analysis.analysis import editable_dpg_value


class GameModelAnalyserManager:
    def __init__(self, agent: GameModelAgent):
        self.agent = agent

    p1_guard_predicted = editable_dpg_value("p1_guard_predicted")
    p2_guard_predicted = editable_dpg_value("p2_guard_predicted")
    p1_position_predicted = editable_dpg_value("p1_position_predicted")
    p2_position_predicted = editable_dpg_value("p2_position_predicted")
    p1_move_predicted = property(
        fget=lambda self: FootsiesMove[dpg.get_value("p1_move_predicted")],
        fset=lambda self, value: dpg.set_value("p1_move_predicted", value.name)
    )
    p2_move_predicted = property(
        fget=lambda self: FootsiesMove[dpg.get_value("p2_move_predicted")],
        fset=lambda self, value: dpg.set_value("p2_move_predicted", value.name)
    )
    p1_move_progress_predicted = editable_dpg_value("p1_move_progress_predicted")
    p2_move_progress_predicted = editable_dpg_value("p2_move_progress_predicted")
    
    agent_action = property(
        fget=lambda self: FootsiesMove[dpg.get_value("agent_action")],
        fset=lambda self, value: dpg.set_value("agent_action", value.name)
    )
    opponent_action = property(
        fget=lambda self: FootsiesMove[dpg.get_value("opponent_action")],
        fset=lambda self, value: dpg.set_value("opponent_action", value.name)
    )

    predicted_time_steps_ahead = editable_dpg_value("predicted_time_steps_ahead")

    def load_predicted_battle_state(self, analyser: Analyser):
        analyser.p1_guard = self.p1_guard_predicted
        analyser.p2_guard = self.p2_guard_predicted
        analyser.p1_position = self.p1_position_predicted
        analyser.p2_position = self.p2_position_predicted
        analyser.p1_move = self.p1_move_predicted
        analyser.p2_move = self.p2_move_predicted
        analyser.p1_move_progress = self.p1_move_progress_predicted
        analyser.p2_move_progress = self.p2_move_progress_predicted
        
        analyser.load_battle_state(analyser.custom_battle_state, require_update=False)

    def add_custom_elements(self, analyser: Analyser):
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
                dpg.add_slider_float(min_value=-4.6, max_value=4.6, tag="p1_position_predicted", enabled=False)
                dpg.add_slider_float(min_value=-4.6, max_value=4.6, tag="p2_position_predicted", enabled=False)

            with dpg.table_row():
                dpg.add_text("Move")
                dpg.add_combo([m.name for m in FOOTSIES_MOVE_INDEX_TO_MOVE], tag="p1_move_predicted", enabled=False)
                dpg.add_combo([m.name for m in FOOTSIES_MOVE_INDEX_TO_MOVE], tag="p2_move_predicted", enabled=False)

            with dpg.table_row():
                dpg.add_text("Move progress")
                dpg.add_slider_float(min_value=0, max_value=1, tag="p1_move_progress_predicted", enabled=False)
                dpg.add_slider_float(min_value=0, max_value=1, tag="p2_move_progress_predicted", enabled=False)
        
        with dpg.group(horizontal=True):
            dpg.add_text("Agent action performed on previous state")
            dpg.add_combo([m.name for m in ActionMap.SIMPLE_ACTIONS], tag="agent_action", callback=lambda: self.predict_next_state(analyser))
        
        with dpg.group(horizontal=True):
            dpg.add_text("Opponent action performed on previous state")
            dpg.add_combo([m.name for m in ActionMap.SIMPLE_ACTIONS], tag="opponent_action", callback=lambda: self.predict_next_state(analyser))

        with dpg.group(horizontal=True):
            dpg.add_text("Predicted time steps ahead:")
            dpg.add_input_float(tag="predicted_time_steps_ahead", enabled=False)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Apply", callback=lambda: self.load_predicted_battle_state(analyser=analyser))

    def update_prediction(self, observation: torch.Tensor, agent_action: int, opponent_action: int):
        next_obs, steps = self.agent.predict(observation, agent_action, opponent_action, 15)
        next_obs = next_obs.squeeze(0)

        p1_move = ActionMap.move_from_move_index(next_obs[2:17].argmax().item())
        p2_move = ActionMap.move_from_move_index(next_obs[17:32].argmax().item())

        self.p1_guard_predicted = round(next_obs[0].item() * 3)
        self.p2_guard_predicted = round(next_obs[1].item() * 3)
        self.p1_move_predicted = p1_move
        self.p2_move_predicted = p2_move
        self.p1_move_progress_predicted = next_obs[32].item()
        self.p2_move_progress_predicted = next_obs[33].item()
        self.p1_position_predicted = next_obs[34].item() * 4.6
        self.p2_position_predicted = next_obs[35].item() * 4.6
        
        self.agent_action = ActionMap.SIMPLE_ACTIONS[agent_action]
        self.opponent_action = ActionMap.SIMPLE_ACTIONS[opponent_action]
        self.predicted_time_steps_ahead = steps

    def predict_next_state(self, observation: torch.Tensor, agent_action: int | None = None, opponent_action: int | None = None):
        # Note: we need to consider the current information when determining the players' moves, but the previous observation!
        agent_action = ActionMap.simple_from_move(self.agent_action) if agent_action is None else agent_action
        opponent_action = ActionMap.simple_from_move(self.opponent_action) if opponent_action is None else opponent_action
        
        self.update_prediction(observation, agent_action, opponent_action)

    def on_state_update(self, analyser: Analyser):
        if analyser.most_recent_transition is not None:
            _, _, _, _, _, info, next_info = analyser.most_recent_transition
            agent_action, opponent_action = ActionMap.simples_from_transition_ori(info, next_info)

            self.predict_next_state(analyser, agent_action, opponent_action)


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

    env = TransformObservation(
        FootsiesActionCombinationsDiscretized(
            FlattenObservation(
                FootsiesNormalized(footsies_env)
            )
        ),
        f=lambda o: torch.from_numpy(o).float().unsqueeze(0)
    )

    agent = GameModelAgent(
        game_model=GameModel(
            game_model_network=GameModelNetwork(
                obs_dim=env.observation_space.shape[0],
                p1_action_dim=ActionMap.n_simple() - 2,
                p2_action_dim=ActionMap.n_simple(),
                hidden_layer_sizes=[64, 64],
                residual=True,
            ),
        ),
        steps_n=[15, 20, 25],
    )

    agent.load("saved/f_opp_recurrent")

    manager = GameModelAnalyserManager(agent)

    analyser = Analyser(
        env=env,
        p1_action_source=lambda obs, info: 0,
        custom_elements_callback=manager.add_custom_elements,
        custom_state_update_callback=manager.on_state_update,
    )
    analyser.start()
