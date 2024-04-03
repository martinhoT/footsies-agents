import copy
import dearpygui.dearpygui as dpg
import pprint
import threading
from datetime import datetime
from typing import Callable
from gymnasium import Env, ObservationWrapper
from gymnasium.wrappers.flatten_observation import FlattenObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.state import FootsiesBattleState, FootsiesState
from footsies_gym.moves import FootsiesMove, FOOTSIES_MOVE_INDEX_TO_MOVE, FOOTSIES_MOVE_ID_TO_INDEX
from agents.action import ActionMap
from dataclasses import dataclass


@dataclass
class AnalyserState:
    battle_state: FootsiesBattleState

    previous_original_observation: dict
    current_original_observation: dict
    previous_observation: "any"
    current_observation: "any"

    previous_info: dict
    current_info: dict
    reward: float
    terminated: bool
    truncated: bool


def editable_dpg_value(item: int | str):
    return property(
        fget=lambda self: dpg.get_value(item),
        fset=lambda self, value: dpg.set_value(item, value),
    )


# NOTE: assumes no other wrapper that transforms the observation space other than observation wrapper is used (like FrameSkipped)
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
        p1_action_source: Callable[["any", dict], "any"],
        custom_elements_callback: Callable[["Analyser"], None], # function that will be called when the main DPG window is being created, allowing the addition of custom elements
        custom_state_update_callback: Callable[["Analyser"], None], # function that will be called when the battle state is updated (either through the 'Advance' button or by manipulation)
    ):
        footsies_env: FootsiesEnv = env.unwrapped
        if footsies_env.sync_mode != "synced_non_blocking":
            raise ValueError("the FOOTSIES environment sync mode is not 'synced_non_blocking', using other modes is not appropriate")
        if footsies_env.render_mode != "human":
            print("WARNING: the environment's render mode was set to a value other than 'human', this might not be intended")
        if footsies_env.by_example:
            print("INFO: player 1 is set to be the in-game bot, and as such custom actions will not be supported")

        self.env = env
        self.footsies_env = footsies_env
        self.p1_action_source = p1_action_source
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

        # Store the current and previous environment information.
        # The current and previous observations are those that are transformed and to be passed to the agent.
        # The original current and previous observations are internal and supposed to be used by the analyser, which shouldn't care about which wrappers are being used.
        self.current_original_observation = None
        self.current_observation = None
        self.current_info = None
        self.previous_original_observation = None
        self.previous_observation = None
        self.previous_info = None

        self.requires_reset = True

        self._custom_battle_state_cached = None

        self.saved_battle_states: list[AnalyserState] = []
        self.episode_counter = -1
        self.wins_counter = 0

        # Allow advance() to be performed continuously on a separate thread
        self.advancing = False

        # DPG items
        self.dpg_saved_battle_state_list: int | str = None
        self.dpg_advance_button: int | str = None
        self.dpg_keep_advancing_button: int | str = None
        self.dpg_stop_advancing_button: int | str = None

        self.text_output_formatter = pprint.PrettyPrinter(indent=1)

    def battle_state_label(self, battle_state: FootsiesBattleState) -> str:
        return f"State at frame {battle_state.frameCount} and episode {self.episode_counter} ({datetime.now().time()})"

    def save_battle_state(self):
        battle_state = self.footsies_env.save_battle_state()
        self.saved_battle_states.append(AnalyserState(
            battle_state=battle_state,
            previous_original_observation=self.previous_original_observation,
            current_original_observation=self.current_original_observation,
            previous_observation=self.previous_observation,
            current_observation=self.current_observation,
            previous_info=self.previous_info,
            current_info=self.current_info,
            reward=self.reward,
            terminated=self.terminated,
            truncated=self.truncated,
        ))

        current_saved_battle_state_list = self.saved_battle_states_labels
        dpg.configure_item(self.dpg_saved_battle_state_list, items=current_saved_battle_state_list + [self.battle_state_label(battle_state)])

    def load_battle_state(self, battle_state: FootsiesBattleState, require_update: bool = True):
        if self.advancing:
            print("WARNING: you are attempting to load a custom battle state while the environment is advancing, you should stop it or multithreading shenanigans happen! We change stuff that is used in advance such as the previous_observation variable. To avoid problems, automatic advancing has been halted")
            self.stop_advancing()

        self.footsies_env.load_battle_state(battle_state)

        if require_update:
            self.p1_position = battle_state.p1State.position[0]
            self.p2_position = battle_state.p2State.position[0]
            self.p1_guard = battle_state.p1State.guardHealth
            self.p2_guard = battle_state.p2State.guardHealth
            self.p1_move = FOOTSIES_MOVE_INDEX_TO_MOVE[FOOTSIES_MOVE_ID_TO_INDEX[battle_state.p1State.currentActionID]]
            self.p2_move = FOOTSIES_MOVE_INDEX_TO_MOVE[FOOTSIES_MOVE_ID_TO_INDEX[battle_state.p2State.currentActionID]]
            self.p1_move_progress = battle_state.p1State.currentActionFrame / self.p1_move.value.duration
            self.p2_move_progress = battle_state.p2State.currentActionFrame / self.p2_move.value.duration
            self.frame = battle_state.frameCount
        
        # TODO: guarantee that p1MostRecentAction is the same as would be normally obtained (it's obtained differently in the game code)
        footsies_state = FootsiesState.from_battle_state(battle_state)
        self.previous_observation = None
        self.current_observation = transformed_observation_from_root(self.env, self.footsies_env._extract_obs(footsies_state))
        self.custom_state_update_callback(self)

    def load_battle_state_from_selected(self):
        saved_state = self.selected_saved_state
        self.load_battle_state(saved_state.battle_state)
        self.update_state(saved_state.current_observation, saved_state.current_original_observation, saved_state.current_info, saved_state.reward, saved_state.terminated, saved_state.truncated,
            previous_original_observation=saved_state.previous_original_observation,
            previous_observation=saved_state.previous_observation,
            previous_info=saved_state.previous_info,
        )

    def update_current_action_from_agent(self):
        action = self.p1_action_source(self.current_observation, self.current_info)
        if action is None:
            raise RuntimeError("this agent could not produce an action for the current observation")

        if self.discretized_actions:
            action_tuple = ActionMap.discrete_to_primitive(action)

        self.action_left, self.action_right, self.action_attack = action_tuple

    def update_state(self, observation: "any", original_observation: dict, info: dict, reward: float, terminated: bool, truncated: bool, *, previous_observation: "any" = None, previous_original_observation: dict = None, previous_info: dict = None):
        # Observation
        self.previous_original_observation = self.current_original_observation if previous_original_observation is None else previous_original_observation
        self.previous_observation = self.current_observation if previous_observation is None else previous_observation
        self.current_original_observation = original_observation
        self.current_observation = observation

        if self.previous_original_observation is not None:
            self.p1_guard_prev = self.previous_original_observation["guard"][0]
            self.p2_guard_prev = self.previous_original_observation["guard"][1]
            self.p1_move_prev = ActionMap.move_from_move_index(self.previous_original_observation["move"][0])
            self.p2_move_prev = ActionMap.move_from_move_index(self.previous_original_observation["move"][1])
            self.p1_move_progress_prev = self.previous_original_observation["move_frame"][0] / self.p1_move_prev.value.duration
            self.p2_move_progress_prev = self.previous_original_observation["move_frame"][1] / self.p2_move_prev.value.duration
            self.p1_position_prev = self.previous_original_observation["position"][0]
            self.p2_position_prev = self.previous_original_observation["position"][1]

        self.p1_guard = self.current_original_observation["guard"][0]
        self.p2_guard = self.current_original_observation["guard"][1]
        self.p1_move = ActionMap.move_from_move_index(self.current_original_observation["move"][0])
        self.p2_move = ActionMap.move_from_move_index(self.current_original_observation["move"][1])
        self.p1_move_progress = self.current_original_observation["move_frame"][0] / self.p1_move.value.duration
        self.p2_move_progress = self.current_original_observation["move_frame"][1] / self.p1_move.value.duration
        self.p1_position = self.current_original_observation["position"][0]
        self.p2_position = self.current_original_observation["position"][1]

        # Info
        self.previous_info = self.current_info if previous_info is None else previous_info
        self.current_info = info
        self.text_output = self.text_output_formatter.pformat(info)
        self.frame = info["frame"]

        # Reward and other stuff
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated

    def start_advancing(self):
        dpg.disable_item(self.dpg_advance_button)
        dpg.disable_item(self.dpg_keep_advancing_button)
        dpg.enable_item(self.dpg_stop_advancing_button)

        self.advancing = True
        while self.advancing:
            self.advance()

    def stop_advancing(self):
        dpg.enable_item(self.dpg_advance_button)
        dpg.enable_item(self.dpg_keep_advancing_button)
        dpg.disable_item(self.dpg_stop_advancing_button)

        self.advancing = False

    def advance(self):
        terminated, truncated = False, False

        if self.requires_reset:
            # We need to reset these variables or else we will report an inter-episode transition which doesn't make sense.
            self.current_info = None
            self.current_observation = None
            self.current_original_observation = None

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

        original_observation = self.footsies_env.most_recent_observation
        self.update_state(obs, original_observation, info, reward, terminated, truncated)
        
        if terminated and reward > 0:
            self.wins_counter += 1
            self.win_rate = self.wins_counter / (self.episode_counter + 1)

        self.custom_state_update_callback(self)
        self._custom_battle_state_cached = None

    @property
    def current_action(self) -> tuple[bool, bool, bool] | int:
        action = (self.action_left, self.action_right, self.action_attack)
        
        if self.discretized_actions:
            return ActionMap.primitive_to_discrete(action)

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
        if self._custom_battle_state_cached is None:
            self._custom_battle_state_cached = copy.copy(self.footsies_env.save_battle_state())

        battle_state = self._custom_battle_state_cached

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

    @property
    def most_recent_transition(self) -> tuple["any", "any", float, bool, bool, dict, dict] | None:
        """The most recent environment transition tuple `(obs, next_obs, reward, terminated, truncated, info, next_info)`. If there hasn't been such a transition yet, return `None`."""
        if self.previous_observation is not None:
            return (self.previous_observation, self.current_observation, self.reward, self.terminated, self.truncated, self.previous_info, self.current_info)
        return None

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
    win_rate: float = editable_dpg_value("win_rate")
    terminated: bool = editable_dpg_value("terminated")
    truncated: bool = editable_dpg_value("truncated")
    text_output: str = editable_dpg_value("text_output")

    def start(self, state_change_apply_immediately: bool = False, debug: bool = False):
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
                    dpg.add_slider_float(min_value=-4.6, max_value=4.6, tag="p1_position_prev", enabled=False)
                    dpg.add_slider_float(min_value=-4.6, max_value=4.6, tag="p2_position_prev", enabled=False)

                with dpg.table_row():
                    dpg.add_text("Move")
                    dpg.add_combo([m.name for m in FOOTSIES_MOVE_INDEX_TO_MOVE], tag="p1_move_prev", enabled=False)
                    dpg.add_combo([m.name for m in FOOTSIES_MOVE_INDEX_TO_MOVE], tag="p2_move_prev", enabled=False)

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
                    dpg.add_slider_int(min_value=0, max_value=3, tag="p1_guard", callback=(lambda: self.load_battle_state(self.custom_battle_state, require_update=False)) if state_change_apply_immediately else None)
                    dpg.add_slider_int(min_value=0, max_value=3, tag="p2_guard", callback=(lambda: self.load_battle_state(self.custom_battle_state, require_update=False)) if state_change_apply_immediately else None)
                
                with dpg.table_row():
                    dpg.add_text("Position")
                    dpg.add_slider_float(min_value=-4.6, max_value=4.6, tag="p1_position", callback=(lambda: self.load_battle_state(self.custom_battle_state, require_update=False)) if state_change_apply_immediately else None)
                    dpg.add_slider_float(min_value=-4.6, max_value=4.6, tag="p2_position", callback=(lambda: self.load_battle_state(self.custom_battle_state, require_update=False)) if state_change_apply_immediately else None)

                with dpg.table_row():
                    dpg.add_text("Move")
                    dpg.add_combo([m.name for m in FOOTSIES_MOVE_INDEX_TO_MOVE], tag="p1_move", callback=(lambda: self.load_battle_state(self.custom_battle_state, require_update=False)) if state_change_apply_immediately else None)
                    dpg.add_combo([m.name for m in FOOTSIES_MOVE_INDEX_TO_MOVE], tag="p2_move", callback=(lambda: self.load_battle_state(self.custom_battle_state, require_update=False)) if state_change_apply_immediately else None)

                with dpg.table_row():
                    dpg.add_text("Move progress")
                    dpg.add_slider_float(min_value=0, max_value=1, tag="p1_move_progress", callback=(lambda: self.load_battle_state(self.custom_battle_state, require_update=False)) if state_change_apply_immediately else None)
                    dpg.add_slider_float(min_value=0, max_value=1, tag="p2_move_progress", callback=(lambda: self.load_battle_state(self.custom_battle_state, require_update=False)) if state_change_apply_immediately else None)

            if not state_change_apply_immediately:
                dpg.add_button(label="Apply", callback=lambda: self.load_battle_state(self.custom_battle_state, require_update=False))

            dpg.add_separator()

            # Footsies battle state manager (save/load)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", callback=self.save_battle_state)
                dpg.add_button(label="Load", callback=self.load_battle_state_from_selected)

                dpg.add_text("ⓘ")
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("Careful when using a dense reward scheme, since the environment internally keeps track of the cumulative reward and that is not reset!\nDon't perform reward-sensitive operations with save/load at the end of episodes in that case.")
            
            self.dpg_saved_battle_state_list = dpg.add_listbox([])

            dpg.add_separator()

            # Action selector
            dpg.add_text("Player 1 action")
            
            with dpg.group(horizontal=True):
                dpg.add_text("Action:")
                # Disabled by default, will simply display what the agent outputs
                dpg.add_checkbox(label="Left", tag="action_left")
                dpg.add_checkbox(label="Right", tag="action_right")
                dpg.add_checkbox(label="Attack", tag="action_attack")
            
            with dpg.group(horizontal=True):
                dpg.add_text("Custom action:")
                dpg.add_checkbox(default_value=False, tag="use_custom_action", enabled=not self.footsies_env.by_example)
            
            dpg.add_separator()

            dpg.add_input_float(label="Reward", tag="reward", enabled=False)
            dpg.add_input_int(label="Frame", tag="frame", enabled=False)
            dpg.add_slider_float(label="Win rate", min_value=0, max_value=1, tag="win_rate", enabled=False)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Terminated", tag="terminated", default_value=False, enabled=False)
                dpg.add_checkbox(label="Truncated", tag="truncated", default_value=False, enabled=False)

            with dpg.group(horizontal=True):
                self.dpg_advance_button = dpg.add_button(label="Advance", callback=self.advance)
                self.dpg_keep_advancing_button = dpg.add_button(label="Keep advancing")
                self.dpg_stop_advancing_button = dpg.add_button(label="Stop advancing", enabled=False)

                dpg.configure_item(self.dpg_keep_advancing_button, callback=lambda: threading.Thread(target=self.start_advancing, args=[], daemon=False).start())
                dpg.configure_item(self.dpg_stop_advancing_button, callback=self.stop_advancing)

            dpg.add_text("Game info (only updated on advance)")
            dpg.add_input_text(multiline=True, no_spaces=False, tag="text_output", enabled=False)

            dpg.add_separator()
            self.custom_elements_callback(self)

        # Create a theme mainly for visible 'disabled' states
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvButton, enabled_state=False):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 50, 50))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 50, 50))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (50, 50, 50))

        dpg.bind_theme(global_theme)

        if debug:
            dpg.show_debug()

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

    analyser = Analyser(env=env, p1_action_source=lambda: 0)
    analyser.start()
