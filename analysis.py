import copy
import dearpygui.dearpygui as dpg
import numpy as np
import pprint
from datetime import datetime
from typing import Callable
from gymnasium import Env, ObservationWrapper
from gymnasium.wrappers.flatten_observation import FlattenObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.state import FootsiesBattleState, FootsiesState
from footsies_gym.moves import FootsiesMove, footsies_move_index_to_move, footsies_move_id_to_index
from agents.base import FootsiesAgentBase
from collections import namedtuple


AnalyserState = namedtuple("AnalyserSavedState", ["battle_state", "observation", "info", "reward"])


def discretized_action_to_tuple(discretized_action: int) -> tuple:
    return ((discretized_action & 1) != 0, (discretized_action & 2) != 0, (discretized_action & 4) != 0)


def tuple_action_to_discretized(tuple_action: tuple) -> int:
    return (tuple_action[0] << 0) + (tuple_action[1] << 1) + (tuple_action[2] << 2)


def footsies_move_from_one_hot(one_hot: np.ndarray) -> FootsiesMove:
    return footsies_move_index_to_move[np.where(one_hot == 1.0)[0].item()]


def editable_dpg_value(item: int | str):
    return property(
        fget=lambda self: dpg.get_value(item),
        fset=lambda self, value: dpg.set_value(item, value),
    )


# NOTE: assumes no other wrapper that transforms the observation space is used (like FrameSkipped)
def transformed_observation_from_root(env: Env, root_obs: "any") -> "any":
    observation_wrappers: list[ObservationWrapper] = []
    current_env = env

    while current_env != current_env.unwrapped:
        if isinstance(current_env, ObservationWrapper):
            observation_wrappers.append(current_env)

        current_env = current_env.env

    obs = root_obs
    for observation_wrapper in reversed(observation_wrappers):
        obs = observation_wrapper.observation(obs)

    return obs


class Analyser:
    
    def __init__(self,
        env: Env,
        agent: FootsiesAgentBase,
        custom_elements_callback: Callable[["Analyser"], None], # function that will be called when the main DPG window is being created, allowing the addition of custom elements
        custom_state_update_callback: Callable[["Analyser"], None], # function that will be called when the battle state is updated (either through the 'Advance' button or by manipulation)
    ):
        footsies_env: FootsiesEnv = env.unwrapped
        if footsies_env.sync_mode != "synced_non_blocking":
            raise ValueError("the FOOTSIES environment sync mode is not 'synced_non_blocking', using other modes is not appropriate")
        if footsies_env.render_mode != "human":
            print("WARNING: the environemnt's render mode was set to a value other than 'human', this might not be intended")

        self.env = env
        self.footsies_env = footsies_env
        self.agent = agent
        self.custom_elements_callback = custom_elements_callback
        self.custom_state_update_callback = custom_state_update_callback

        # Check environment wrappers
        self.discretized_actions = False
        normalized_observations = False
        current_env = env
        while current_env != self.footsies_env:
            if isinstance(current_env, FootsiesActionCombinationsDiscretized):
                self.discretized_actions = True

            if isinstance(current_env, FootsiesNormalized):
                normalized_observations = True
    
            current_env = current_env.env

        if not normalized_observations:
            raise ValueError("the environment should have the normalized observations wrapper on")

        self.current_observation = None
        self.current_info = None
        self.previous_observation = None
        self.previous_info = None
        self.requires_reset = True

        self.saved_battle_states: list[AnalyserState] = []
        self.episode_counter = -1

        # DPG items
        self.dpg_saved_battle_state_list: int | str = None

        self.text_output_formatter = pprint.PrettyPrinter(indent=1)

    def battle_state_label(self, battle_state: FootsiesBattleState) -> str:
        return f"State at frame {battle_state.frameCount} and episode {self.episode_counter} ({datetime.now().time()})"

    def save_battle_state(self):
        battle_state = self.footsies_env.save_battle_state()
        self.saved_battle_states.append(AnalyserState(
            battle_state, self.current_observation, self.current_info, self.reward
        ))

        current_saved_battle_state_list = self.saved_battle_states_labels
        dpg.configure_item(self.dpg_saved_battle_state_list, items=current_saved_battle_state_list + [self.battle_state_label(battle_state)])

    def load_battle_state(self, battle_state: FootsiesBattleState, require_update: bool = True):
        self.footsies_env.load_battle_state(battle_state)

        if require_update:
            self.p1_position = battle_state.p1State.position[0]
            self.p2_position = battle_state.p2State.position[0]
            self.p1_guard = battle_state.p1State.guardHealth
            self.p2_guard = battle_state.p2State.guardHealth
            self.p1_move = footsies_move_index_to_move[footsies_move_id_to_index[battle_state.p1State.currentActionID]]
            self.p2_move = footsies_move_index_to_move[footsies_move_id_to_index[battle_state.p2State.currentActionID]]
            self.p1_move_progress = battle_state.p1State.currentActionFrame / self.p1_move.value.duration
            self.p2_move_progress = battle_state.p2State.currentActionFrame / self.p2_move.value.duration
            self.frame = battle_state.frameCount
        
        # TODO: guarantee that p1MostRecentAction is the same as would be normally obtained (it's obtained differently in the game code)
        footsies_state = FootsiesState.from_battle_state(battle_state)
        self.previous_observation = self.current_observation
        self.current_observation = transformed_observation_from_root(self.env, self.footsies_env._extract_obs(footsies_state))
        self.custom_state_update_callback(self)

    def load_battle_state_from_selected(self):
        saved_state = self.selected_saved_state
        self.load_battle_state(saved_state.battle_state)
        self.update_state(saved_state.observation, saved_state.info, saved_state.reward)

    def update_current_action_from_agent(self):
        action = self.agent.act(self.current_observation)
        if action is None:
            raise RuntimeError("this agent could not produce an action for the current observation")

        if self.discretized_actions:
            action_tuple = discretized_action_to_tuple(action)

        self.action_left, self.action_right, self.action_attack = action_tuple

    def update_state(self, observation, info, reward):
        # Observation
        self.previous_observation = self.current_observation
        self.current_observation = observation

        if self.previous_observation is not None:
            self.p1_guard_prev = round(self.previous_observation[0] * 3)
            self.p2_guard_prev = round(self.previous_observation[1] * 3)
            self.p1_move_prev = footsies_move_from_one_hot(self.previous_observation[2:17])
            self.p2_move_prev = footsies_move_from_one_hot(self.previous_observation[17:32])
            self.p1_move_progress_prev = self.previous_observation[32]
            self.p2_move_progress_prev = self.previous_observation[33]
            self.p1_position_prev = self.previous_observation[34] * 4.4
            self.p2_position_prev = self.previous_observation[35] * 4.4

        self.p1_guard = round(self.current_observation[0] * 3)
        self.p2_guard = round(self.current_observation[1] * 3)
        self.p1_move = footsies_move_from_one_hot(self.current_observation[2:17])
        self.p2_move = footsies_move_from_one_hot(self.current_observation[17:32])
        self.p1_move_progress = self.current_observation[32]
        self.p2_move_progress = self.current_observation[33]
        self.p1_position = self.current_observation[34] * 4.4
        self.p2_position = self.current_observation[35] * 4.4

        # Info
        self.previous_info = self.current_info
        self.current_info = info
        self.text_output = self.text_output_formatter.pformat(info)
        self.frame = info["frame"]

        # Reward
        self.reward = reward

    def advance(self):
        if self.requires_reset:
            self.episode_counter += 1
            obs, info = self.env.reset()

            reward = 0
            self.requires_reset = False

        else:
            if not self.use_custom_action:
                self.update_current_action_from_agent()

            current_action = self.current_action
            obs, reward, terminated, truncated, info = self.env.step(current_action)

            self.requires_reset = terminated or truncated

        self.update_state(obs, info, reward)
        
        self.custom_state_update_callback(self)

    @property
    def current_action(self) -> tuple[bool, bool, bool] | int:
        action = (self.action_left, self.action_right, self.action_attack)
        
        if self.discretized_actions:
            return tuple_action_to_discretized(action)

        return action

    @property
    def saved_battle_states_labels(self) -> list[str]:
        return dpg.get_item_configuration(self.dpg_saved_battle_state_list)["items"]

    @property
    def selected_saved_state(self) -> AnalyserState:
        selected_battle_state_label = dpg.get_value(self.dpg_saved_battle_state_list)
        index = self.saved_battle_states_labels.index(selected_battle_state_label)
        return self.saved_battle_states[index]

    @property
    def custom_battle_state(self) -> FootsiesBattleState:
        battle_state = copy.copy(self.footsies_env.save_battle_state())
        
        battle_state.p1State.position[0] = self.p1_position
        battle_state.p2State.position[0] = self.p2_position
        battle_state.p1State.guardHealth = self.p1_guard
        battle_state.p2State.guardHealth = self.p2_guard
        battle_state.p1State.currentActionID = self.p1_move.value.id
        battle_state.p2State.currentActionID = self.p2_move.value.id
        battle_state.p1State.currentActionFrame = round(self.p1_move_progress * self.p1_move.value.duration)
        battle_state.p2State.currentActionFrame = round(self.p2_move_progress * self.p2_move.value.duration)
        battle_state.frameCount = self.frame

        return battle_state

    p1_guard_prev: int = editable_dpg_value("p1_guard_prev")
    p2_guard_prev: int = editable_dpg_value("p2_guard_prev")
    p1_position_prev: float = editable_dpg_value("p1_position_prev")
    p2_position_prev: float = editable_dpg_value("p2_position_prev")
    p1_move_prev: FootsiesMove = property(
        fget=lambda self: FootsiesMove[dpg.get_value("p1_move_prev")],
        fset=lambda self, value: dpg.set_value("p1_move_prev", value.name)
    )
    p2_move_prev: FootsiesMove = property(
        fget=lambda self: FootsiesMove[dpg.get_value("p2_move_prev")],
        fset=lambda self, value: dpg.set_value("p2_move_prev", value.name)
    )
    p1_move_progress_prev: float = editable_dpg_value("p1_move_progress_prev")
    p2_move_progress_prev: float = editable_dpg_value("p2_move_progress_prev")

    p1_guard: int = editable_dpg_value("p1_guard")
    p2_guard: int = editable_dpg_value("p2_guard")
    p1_position: float = editable_dpg_value("p1_position")
    p2_position: float = editable_dpg_value("p2_position")
    p1_move: FootsiesMove = property(
        fget=lambda self: FootsiesMove[dpg.get_value("p1_move")],
        fset=lambda self, value: dpg.set_value("p1_move", value.name)
    )
    p2_move: FootsiesMove = property(
        fget=lambda self: FootsiesMove[dpg.get_value("p2_move")],
        fset=lambda self, value: dpg.set_value("p2_move", value.name)
    )
    p1_move_progress: float = editable_dpg_value("p1_move_progress")
    p2_move_progress: float = editable_dpg_value("p2_move_progress")

    action_left: bool = editable_dpg_value("action_left")
    action_right: bool = editable_dpg_value("action_right")
    action_attack: bool = editable_dpg_value("action_attack")
    use_custom_action: bool = editable_dpg_value("use_custom_action")
    reward: float = editable_dpg_value("reward")
    frame: int = editable_dpg_value("frame")
    text_output: str = editable_dpg_value("text_output")

    def start(self):
        dpg.create_context()
        dpg.create_viewport(title="FOOTSIES data analyser", width=600, height=300)

        with dpg.window() as main_window:
            # Footsies previous battle state
            dpg.add_text("Previous state")
            with dpg.table():
                dpg.add_table_column(label="Property")
                dpg.add_table_column(label="P1")
                dpg.add_table_column(label="P2")

                with dpg.table_row():
                    dpg.add_text("Guard")
                    dpg.add_slider_int(min_value=0, max_value=3, tag="p1_guard_prev", enabled=False)
                    dpg.add_slider_int(min_value=0, max_value=3, tag="p2_guard_prev", enabled=False)
                
                with dpg.table_row():
                    dpg.add_text("Position")
                    dpg.add_slider_float(min_value=-4.4, max_value=4.4, tag="p1_position_prev", enabled=False)
                    dpg.add_slider_float(min_value=-4.4, max_value=4.4, tag="p2_position_prev", enabled=False)

                with dpg.table_row():
                    dpg.add_text("Move")
                    dpg.add_combo([m.name for m in footsies_move_index_to_move], tag="p1_move_prev", enabled=False)
                    dpg.add_combo([m.name for m in footsies_move_index_to_move], tag="p2_move_prev", enabled=False)

                with dpg.table_row():
                    dpg.add_text("Move progress")
                    dpg.add_slider_float(min_value=0, max_value=1, tag="p1_move_progress_prev", enabled=False)
                    dpg.add_slider_float(min_value=0, max_value=1, tag="p2_move_progress_prev", enabled=False)

            dpg.add_separator()    
            
            # Footsies battle state modifier
            dpg.add_text("Current state")
            with dpg.table():
                dpg.add_table_column(label="Property")
                dpg.add_table_column(label="P1")
                dpg.add_table_column(label="P2")

                with dpg.table_row():
                    dpg.add_text("Guard")
                    dpg.add_slider_int(min_value=0, max_value=3, tag="p1_guard")
                    dpg.add_slider_int(min_value=0, max_value=3, tag="p2_guard")
                
                with dpg.table_row():
                    dpg.add_text("Position")
                    dpg.add_slider_float(min_value=-4.4, max_value=4.4, tag="p1_position")
                    dpg.add_slider_float(min_value=-4.4, max_value=4.4, tag="p2_position")

                with dpg.table_row():
                    dpg.add_text("Move")
                    dpg.add_combo([m.name for m in footsies_move_index_to_move], tag="p1_move")
                    dpg.add_combo([m.name for m in footsies_move_index_to_move], tag="p2_move")

                with dpg.table_row():
                    dpg.add_text("Move progress")
                    dpg.add_slider_float(min_value=0, max_value=1, tag="p1_move_progress")
                    dpg.add_slider_float(min_value=0, max_value=1, tag="p2_move_progress")

            dpg.add_button(label="Apply", callback=lambda: self.load_battle_state(self.custom_battle_state, require_update=False))

            dpg.add_separator()

            # Footsies battle state manager (save/load)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", callback=self.save_battle_state)
                dpg.add_button(label="Load", callback=self.load_battle_state_from_selected)
            
            self.dpg_saved_battle_state_list = dpg.add_listbox([])

            dpg.add_separator()

            # Action selector
            with dpg.group(horizontal=True):
                dpg.add_text("Action:")
                # Disabled by default, will simply display what the agent outputs
                dpg.add_checkbox(label="Left", tag="action_left")
                dpg.add_checkbox(label="Right", tag="action_right")
                dpg.add_checkbox(label="Attack", tag="action_attack")
            
            with dpg.group(horizontal=True):
                dpg.add_text("Custom action:")
                dpg.add_checkbox(default_value=False, tag="use_custom_action")
            
            dpg.add_separator()

            dpg.add_input_float(label="Reward", tag="reward", enabled=False)
            dpg.add_input_int(label="Frame", tag="frame", enabled=False)

            dpg.add_button(label="Advance", callback=self.advance)

            dpg.add_text("Game info (only updated on advance)")
            dpg.add_input_text(multiline=True, no_spaces=False, tag="text_output", enabled=False)

            dpg.add_separator()
            self.custom_elements_callback(self)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(main_window, True)
        dpg.start_dearpygui()
        dpg.destroy_context()


if __name__ == "__main__":

    footsies_env = FootsiesEnv(
        game_path="../Footsies-Gym/Build/FOOTSIES.x86_64",
        game_port=15000,
        opponent_port=15001,
        render_mode="human",
        sync_mode="synced_non_blocking",
        fast_forward=False,

        log_file="out.log",
        log_file_overwrite=True,
    )

    env = FootsiesActionCombinationsDiscretized(
        FlattenObservation(
            FootsiesNormalized(footsies_env)
        )
    )

    analyser = Analyser(env=env, agent=None)
    analyser.start()
