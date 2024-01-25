import dearpygui as dpg
import copy
from gymnasium import Env
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.state import FootsiesBattleState
from footsies_gym.moves import footsies_move_id_to_index, footsies_move_index_to_move, FootsiesMove
from agents.base import FootsiesAgentBase


def discretized_action_to_tuple(discretized_action):
    return ((discretized_action & 1) != 0, (discretized_action & 2) != 0, (discretized_action & 4) != 0)


def footsies_move(move_id: int) -> FootsiesMove:
    return footsies_move_index_to_move(footsies_move_id_to_index(move_id))


class Analyser:
    
    def __init__(self,
        env: Env,
        agent: FootsiesAgentBase,
    ):
        footsies_env = env.unwrapped
        if footsies_env.sync_mode != "synced_non_blocking":
            raise ValueError("the FOOTSIES environment sync mode is not 'synced_non_blocking', using other modes is not appropriate")

        self.env = env
        self.footsies_env = footsies_env
        self.agent = agent

        # Check environment wrappers
        self.discretized_actions = False
        normalized_observations = False
        current_env = env
        while current_env != self.footsies_env:
            if isinstance(current_env, FootsiesActionCombinationsDiscretized):
                self.discretized_actions = True

            if isinstance(current_env, FootsiesNormalized):
                normalized_observations = True
    
            current_env = current_env.unwrapped

        if not normalized_observations:
            raise ValueError("the environment should have the normalized observations wrapper on")

        self.current_observation = None

        self.saved_battle_states = []
        self.previous_state = None
        self.next_state = None

        self.dpg_saved_battle_state_list = None
        self.dpg_battle_state_p1_guard = None
        self.dpg_battle_state_p2_guard = None
        self.dpg_battle_state_p1_position = None
        self.dpg_battle_state_p2_position = None
        self.dpg_battle_state_p1_move = None
        self.dpg_battle_state_p2_move = None
        self.dpg_battle_state_p1_move_progress = None
        self.dpg_battle_state_p2_move_progress = None
        self.dpg_action_left_button = None
        self.dpg_action_right_button = None
        self.dpg_action_attack_button = None
        self.dpg_reward_display = None

    def battle_state_label(self, battle_state: FootsiesBattleState) -> str:
        return f"State at frame {battle_state.frameCount}"

    def save_battle_state(self):
        battle_state = self.env.save_battle_state()
        self.saved_battle_states.append(battle_state)

        current_saved_battle_state_list = self.saved_battle_states_labels
        dpg.configure_item(self.dpg_saved_battle_state_list, items=current_saved_battle_state_list + [self.battle_state_label(battle_state)])

    def load_battle_state(self, battle_state: FootsiesBattleState):
        self.env.load_battle_state(battle_state)

    def load_battle_state_from_selected(self):
        self.load_battle_state(self.selected_battle_state)

    def update_current_action_from_agent(self):
        action = self.agent.act(self.current_observation)
        if self.discretized_actions:
            action_tuple = discretized_action_to_tuple(action)

        dpg.set_value(self.dpg_action_left_button, action_tuple[0])
        dpg.set_value(self.dpg_action_right_button, action_tuple[1])
        dpg.set_value(self.dpg_action_attack_button, action_tuple[2])

    def update_reward(self, reward: float):
        dpg.set_value(self.dpg_reward_display, reward)

    # TODO: logic
    def advance(self):
        
        self.env.reset()

        if not self.use_custom_action:
            self.update_current_action_from_agent()

        current_action = self.current_action()
        next_observation, reward, terminated, truncated, info = self.env.step(current_action)

        self.update_reward(reward)

        if terminated or truncated:
            pass

    @property
    def current_action(self) -> tuple[bool, bool, bool]:
        return (
            dpg.get_value(self.dpg_action_left_button),
            dpg.get_value(self.dpg_action_right_button),
            dpg.get_value(self.dpg_action_attack_button),
        )

    @property
    def use_custom_action(self) -> bool:
        # Default value
        if self.dpg_custom_action_checkbox is None:
            return False
        
        return dpg.get_value(self.dpg_custom_action_checkbox)

    @property
    def saved_battle_states_labels(self) -> list[str]:
        return dpg.get_item_configuration()["items"]

    @property
    def selected_battle_state(self) -> FootsiesBattleState:
        selected_battle_state_label = dpg.get_value(self.dpg_saved_battle_state_list)
        index = self.saved_battle_states_labels.index(selected_battle_state_label)
        return self.saved_battle_states[index]

    @property
    def custom_battle_state(self) -> FootsiesBattleState:
        battle_state = copy.copy(self.selected_battle_state)
        
        battle_state.p1State.position[0] = self.p1_position
        battle_state.p2State.position[0] = self.p2_position
        battle_state.p1State.guardHealth = self.p1_guard
        battle_state.p2State.guardHealth = self.p2_guard
        battle_state.p1State.currentActionID = self.p1_move
        battle_state.p2State.currentActionID = self.p2_move
        battle_state.p1State.currentActionFrame = round(self.p1_move_progress * footsies_move(self.p1_move).value.duration)
        battle_state.p2State.currentActionFrame = round(self.p2_move_progress * footsies_move(self.p2_move).value.duration)

    @property
    def p1_guard(self) -> int:
        return dpg.get_value(self.dpg_battle_state_p1_guard)

    @property
    def p2_guard(self) -> int:
        return dpg.get_value(self.dpg_battle_state_p2_guard)
    
    @property
    def p1_position(self) -> int:
        return dpg.get_value(self.dpg_battle_state_p1_position)
    
    @property
    def p2_position(self) -> int:
        return dpg.get_value(self.dpg_battle_state_p2_position)
    
    @property
    def p1_move(self) -> int:
        return dpg.get_value(self.dpg_battle_state_p1_move)
    
    @property
    def p2_move(self) -> int:
        return dpg.get_value(self.dpg_battle_state_p2_move)
    
    @property
    def p1_move_progress(self) -> int:
        return dpg.get_value(self.dpg_battle_state_p1_move_progress)
    
    @property
    def p2_move_progress(self) -> int:
        return dpg.get_value(self.dpg_battle_state_p2_move_progress)

    def start(self):
        dpg.create_context()
        dpg.create_viewport(title="FOOTSIES data analyser", width=600, height=300)

        with dpg.window() as main_window:
            # Battle state zone
            with dpg.group(horizontal=True):
                # Footsies battle state modifier
                with dpg.group():
                    with dpg.table():
                        dpg.add_table_column(label="Property")
                        dpg.add_table_column(label="P1")
                        dpg.add_table_column(label="P2")

                        with dpg.table_row():
                            dpg.add_text("Guard")
                            self.dpg_battle_state_p1_guard = dpg.add_int_value()
                            self.dpg_battle_state_p2_guard = dpg.add_int_value()
                        
                        with dpg.table_row():
                            dpg.add_text("Position")
                            self.dpg_battle_state_p1_position = dpg.add_float_value()
                            self.dpg_battle_state_p2_position = dpg.add_float_value()

                        with dpg.table_row():
                            dpg.add_text("Move")
                            # TODO: use dropdowns instead
                            self.dpg_battle_state_p1_move = dpg.add_int_value()
                            self.dpg_battle_state_p2_move = dpg.add_int_value()

                        with dpg.table_row():
                            dpg.add_text("Move progress")
                            self.dpg_battle_state_p1_move_progress = dpg.add_float_value()
                            self.dpg_battle_state_p2_move_progress = dpg.add_float_value()

                    dpg.add_button(label="Apply")

                # Footsies battle state manager (save/load)
                with dpg.group():
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Save", callback=self.save_battle_state)
                        dpg.add_button(label="Load", callback=self.load_battle_state)
                    
                    self.dpg_saved_battle_state_list = dpg.add_listbox([])

            # Action selector
            with dpg.group(horizontal=True):
                dpg.add_text("Action:")
                # Disabled by default, will simply display what the agent outputs
                self.dpg_action_left_button = dpg.add_checkbox(enabled=False)
                self.dpg_action_right_button = dpg.add_checkbox(enabled=False)
                self.dpg_action_attack_button = dpg.add_checkbox(enabled=False)
            
            with dpg.group(horizontal=True):
                dpg.add_text("Custom action:")
                self.dpg_custom_action_checkbox = dpg.add_checkbox(default_value=self.use_custom_action)
            
            self.dpg_reward_display = dpg.add_float_value(label="Reward")

            dpg.add_button(label="Advance", callback=self.advance)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(main_window, True)
        dpg.start_dearpygui()
        dpg.destroy_context()