from typing import cast
import torch as T
import dearpygui.dearpygui as dpg
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.moves import FootsiesMove, FOOTSIES_MOVE_INDEX_TO_MOVE
from agents.game_model.game_model import GameModel, GameModelNetwork
from scripts.analysis.base import Analyser
from agents.game_model.agent import GameModelAgent
from agents.action import ActionMap
from scripts.analysis.base import editable_dpg_value


class GameModelAnalyserManager:
    def __init__(self, agent: GameModelAgent):
        self.agent = agent

        self._observation: T.Tensor | None = None
        self._agent_action: int | None = None
        self._opponent_action: int | None = None

        # DPG items
        self._complete_prediction_group: int | str = ""
        self._specific_prediction_group: int | str = ""

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

    complete_prediction_method = editable_dpg_value("complete_prediction_method")
    selected_game_model = cast(
        tuple[int, GameModel],
        property(
            fget=lambda self: self._get_game_model_dpg(),
            fset=lambda self, value: dpg.set_value("selected_game_model", str(value[0]))
        )
    )
    predict_time_steps_ahead = editable_dpg_value("predict_time_steps_ahead")
    predicted_time_steps_ahead = editable_dpg_value("predicted_time_steps_ahead")

    def _get_game_model_dpg(self) -> tuple[int, GameModel]:
        n = int(dpg.get_value("selected_game_model"))
        model_list = [(step_n, g) for step_n, g in self.agent.game_models if step_n == n]
        assert len(model_list) > 0
        return model_list[0]

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

    def apply_and_advance(self, analyser: Analyser):
        self.load_predicted_battle_state(analyser)
        analyser.advance()

    def add_custom_elements(self, analyser: Analyser):
        dpg.add_text("Predicted current state based on the previous observation")
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
            dpg.add_combo([m.name for m in ActionMap.SIMPLE_ACTIONS], tag="agent_action", callback=lambda: self.predict_next_state())
        
        with dpg.group(horizontal=True):
            dpg.add_text("Opponent action performed on previous state")
            dpg.add_combo([m.name for m in ActionMap.SIMPLE_ACTIONS], tag="opponent_action", callback=lambda: self.predict_next_state())

        dpg.add_checkbox(label="Complete prediction", default_value=True, tag="complete_prediction_method", callback=self._toggle_prediction_method)

        with dpg.group() as self._complete_prediction_group:
            with dpg.group(horizontal=True):
                dpg.add_text("Predict time steps ahead:")
                dpg.add_input_int(tag="predict_time_steps_ahead", min_value=self.agent.min_resolution, callback=lambda: self.predict_next_state())

        with dpg.group() as self._specific_prediction_group:
            dpg.add_text("Game model (in terms of steps predicted):")
            dpg.add_radio_button([str(step_n) for step_n, _ in self.agent.game_models], tag="selected_game_model", indent=8, default_value=str(self.agent.game_models[0][0]), callback=lambda: self.predict_next_state())

        dpg.hide_item(self._specific_prediction_group)

        with dpg.group(horizontal=True):
            dpg.add_text("Predicted time steps ahead:")
            dpg.add_input_int(tag="predicted_time_steps_ahead", enabled=False)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Apply", callback=lambda: self.load_predicted_battle_state(analyser=analyser))
            dpg.add_button(label="Apply and advance", callback=lambda: self.apply_and_advance(analyser=analyser))

    def _toggle_prediction_method(self):
        if self.complete_prediction_method:
            dpg.show_item(self._complete_prediction_group)
            dpg.hide_item(self._specific_prediction_group)
        else:
            dpg.hide_item(self._complete_prediction_group)
            dpg.show_item(self._specific_prediction_group)

    def predict_next_state(self, observation: T.Tensor | None = None, agent_action: int | None = None, opponent_action: int | None = None):
        # NOTE: we need to consider the current information when determining the players' moves, but the previous observation!
        if observation is None:
            if self._observation is None:
                raise ValueError("cannot perform a prediction yet")
            observation = self._observation
        agent_action = ActionMap.simple_from_move(self.agent_action) if agent_action is None else agent_action
        opponent_action = ActionMap.simple_from_move(self.opponent_action) if opponent_action is None else opponent_action
        
        self.agent_action = ActionMap.SIMPLE_ACTIONS[agent_action]
        self.opponent_action = ActionMap.SIMPLE_ACTIONS[opponent_action]

        if self.complete_prediction_method:
            try:
                next_obs, steps = self.agent.predict(observation, agent_action, opponent_action, self.predict_time_steps_ahead)
            except ValueError:
                return
        else:
            steps, game_model = self.selected_game_model
            next_obs = game_model.predict(observation, agent_action ,opponent_action)
        next_obs = next_obs.squeeze(0)

        p1_move_index = int(next_obs[2:17].argmax().item())
        p2_move_index = int(next_obs[17:32].argmax().item())
        p1_move = ActionMap.move_from_move_index(p1_move_index)
        p2_move = ActionMap.move_from_move_index(p2_move_index)

        self.p1_guard_predicted = round(next_obs[0].item() * 3)
        self.p2_guard_predicted = round(next_obs[1].item() * 3)
        self.p1_move_predicted = p1_move
        self.p2_move_predicted = p2_move
        self.p1_move_progress_predicted = next_obs[32].item()
        self.p2_move_progress_predicted = next_obs[33].item()
        self.p1_position_predicted = next_obs[34].item() * 4.6
        self.p2_position_predicted = next_obs[35].item() * 4.6
        
        self.predicted_time_steps_ahead = steps

    def set_state(self, observation: T.Tensor, agent_action: int | None = None, opponent_action: int | None = None):
        self._observation = observation
        if agent_action is not None:
            self._agent_action = agent_action
        if opponent_action is not None:
            self._opponent_action = opponent_action

    def on_state_update(self, analyser: Analyser):
        if analyser.most_recent_transition is not None:
            obs, _, _, _, _, info, next_info = analyser.most_recent_transition.as_tuple()
            agent_action, opponent_action = ActionMap.simples_from_transition_ori(info, next_info)

            self.set_state(obs, agent_action, opponent_action)
            self.predict_next_state(obs, agent_action, opponent_action)


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
        f=lambda o: T.from_numpy(o).float().unsqueeze(0)
    )
    assert env.observation_space.shape

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
