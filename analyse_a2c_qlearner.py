import torch
import dearpygui.dearpygui as dpg
import numpy as np
from gymnasium.wrappers.flatten_observation import FlattenObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.moves import FootsiesMove
from analysis import Analyser
from agents.action import ActionMap
from agents.a2c.agent import FootsiesAgent as A2CAgent
from agents.ql.ql import QFunctionTable
from main import load_agent_model


SIMPLE_ACTION_LABELS_GEN = lambda labels, n: tuple((move.name, i / n + (1 / (n * 2))) for i, move in enumerate(labels))
SIMPLE_ACTION_LABELS_REVERSED_GEN = lambda labels, n: tuple((move.name, (n - 1) / n - i / n + (1 / (n * 2))) for i, move in enumerate(labels))

SIMPLE_ACTION_LABELS = SIMPLE_ACTION_LABELS_GEN(ActionMap.SIMPLE_ACTIONS, ActionMap.n_simple())
SIMPLE_ACTION_LABELS_REVERSED = SIMPLE_ACTION_LABELS_REVERSED_GEN(ActionMap.SIMPLE_ACTIONS, ActionMap.n_simple())


class QTablePlot:
    def __init__(
        self,
        action_dim: int,
        opponent_action_dim: int,
        add_color_scale: bool = True,
        auto_scale: bool = False,
    ):
        self.action_dim = action_dim
        self.opponent_action_dim = opponent_action_dim
        self.add_color_scale = add_color_scale
        self.auto_scale = auto_scale
        self.color_scale = None
        self.series = None

    def setup(self, title: str = "Q-table"):
        with dpg.group(horizontal=True):
            if self.add_color_scale:
                self.color_scale = dpg.add_colormap_scale(min_scale=-1.0, max_scale=1.0, colormap=dpg.mvPlotColormap_Viridis, height=400)
            with dpg.plot(label=title, no_mouse_pos=True, height=400, width=-1) as plot: # height=400, width=-1
                dpg.bind_colormap(plot, dpg.mvPlotColormap_Viridis)
                x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Agent action", lock_min=True, lock_max=True, no_gridlines=True, no_tick_marks=True)
                dpg.set_axis_ticks(x_axis, SIMPLE_ACTION_LABELS_GEN(ActionMap.SIMPLE_ACTIONS[:self.action_dim], self.action_dim))
                with dpg.plot_axis(dpg.mvYAxis, label="Opponent action", lock_min=True, lock_max=True, no_gridlines=True, no_tick_marks=True) as y_axis:
                    dpg.set_axis_ticks(y_axis, SIMPLE_ACTION_LABELS_REVERSED_GEN(ActionMap.SIMPLE_ACTIONS[:self.opponent_action_dim], self.opponent_action_dim))
                    initial_data = np.zeros((self.opponent_action_dim, self.action_dim))
                    self.series = dpg.add_heat_series(initial_data.flatten().tolist(), rows=self.opponent_action_dim, cols=self.action_dim, scale_min=-1.0, scale_max=1.0)

    def update(self, q_values: np.ndarray):
        dpg.set_value(self.series, [q_values.flatten().tolist(), [self.opponent_action_dim, self.action_dim]])
        if self.auto_scale:
            mn, mx = np.min(q_values), np.max(q_values)
            if self.color_scale is not None:
                dpg.configure_item(self.color_scale, min_scale=mn, max_scale=mx)
            dpg.configure_item(self.series, scale_min=mn, scale_max=mx)


class PolicyDistributionPlot:
    def __init__(
        self,
        action_dim: int,
        bar_width: int = 0.2,
    ):
        self.bar_width = bar_width
        
        self.action_dim = action_dim
        self.x = np.arange(action_dim)

        # DPG items
        self.x_axis = None
        self.y_axis = None
        self.series = None
    
    def setup(self, title: str = "Policy distribution", width: int = 1050):
        y = np.zeros((self.action_dim,))

        with dpg.plot(label=title, width=width) as plot:
            self.x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Move")
            dpg.set_axis_ticks(self.x_axis, tuple([(move.name, i + 1) for i, move in enumerate(ActionMap.SIMPLE_ACTIONS[:self.action_dim])]))
            dpg.set_axis_limits(self.x_axis, 0.0, self.action_dim + 1) # the + 1 is padding

            self.y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Probability")
            dpg.set_axis_limits(self.y_axis, 0.0, 1.0)
        
            self.series = dpg.add_bar_series(self.x + 1.0, y, weight=self.bar_width * 2, parent=self.y_axis)
            dpg.bind_colormap(plot, dpg.mvPlotColormap_Default)

    def update(self, distribution: np.ndarray):
        dpg.set_value(self.series, [list(self.x + 1.0), distribution.tolist()])


class QLearnerAnalyserManager:
    def __init__(
        self,
        agent: A2CAgent,
        action_dim: int,
        opponent_action_dim: int,
    ):
        self.agent = agent
        self.q_table = None
        self.action_dim = action_dim
        self.opponent_action_dim = opponent_action_dim

        self.q_table_plot = QTablePlot(action_dim, opponent_action_dim, auto_scale=False)
        self.q_table_update_frequency_plot = QTablePlot(action_dim, opponent_action_dim, add_color_scale=False, auto_scale=True) if isinstance(self.agent.critic, QFunctionTable) else None
        self.policy_plot = PolicyDistributionPlot(action_dim)
        
        self.current_observation = None

        # DPG items
        self.predicted_opponent_action = None

    def add_custom_elements(self, analyser: Analyser):
        self.q_table_plot.setup(title="Q-table values")
    
        if self.q_table_update_frequency_plot is not None:
            self.q_table_update_frequency_plot.setup(title="Q-table update frequency")        

        dpg.add_separator()
        
        self.policy_plot.setup()
        
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_text("Predicted opponent action:")
            self.predicted_opponent_action = dpg.add_combo([m.name for m in ActionMap.SIMPLE_ACTIONS], default_value=FootsiesMove.STAND.name, enabled=True, callback=lambda s: self.update_policy_distribution())

    def update_policy_distribution(self, obs: np.ndarray = None):
        if obs is None:
            obs = self.current_observation
        
        opponent_action = self.get_predicted_opponent_action()
        policy_distribution = self.agent.actor.probabilities(obs, opponent_action).detach().numpy().flatten()
        self.policy_plot.update(policy_distribution)

    def on_state_update(self, analyser: Analyser):
        obs: torch.Tensor = analyser.current_observation
        self.current_observation = obs

        q_values = self.agent.critic.q(obs)
        self.q_table_plot.update(q_values)
        if self.q_table_update_frequency_plot is not None:
            self.q_table_update_frequency_plot.update(self.agent.critic.update_frequency(obs))

        self.update_policy_distribution(obs)

    def get_predicted_opponent_action(self):
        return ActionMap.simple_from_move(FootsiesMove[dpg.get_value(self.predicted_opponent_action)])


if __name__ == "__main__":

    footsies_env = FootsiesEnv(
        game_path="../Footsies-Gym/Build/FOOTSIES.x86_64",
        game_port=15000,
        opponent_port=15001,
        remote_control_port=15002,
        render_mode="human",
        sync_mode="synced_non_blocking",
        fast_forward=False,
        vs_player=True,
    )

    env = FootsiesActionCombinationsDiscretized(
        FlattenObservation(
            FootsiesNormalized(footsies_env)
        )
    )

    agent = A2CAgent(
        observation_space_size=env.observation_space.shape[0],
        action_space_size=env.action_space.n,
        use_simple_actions=True,
        use_q_table=True,
        use_q_network=False,
        consider_opponent_action=True,
        actor_hidden_layer_sizes_specification="64,64",
        actor_hidden_layer_activation_specification="ReLU",
        critic_hidden_layer_sizes_specification="128,128",
        critic_hidden_layer_activation_specification="ReLU",
        **{
            "a2c.policy_cumulative_discount": False,
            "critic.discount": 1.0,
            "critic.learning_rate": 0.001,
            "a2c.actor_entropy_loss_coef": 0.1,
            "a2c.actor_optimizer.lr": 0.001,
        }
    )

    load_agent_model(agent, "a2c_qlearner_il")

    def spammer():
        from itertools import cycle
        for action in cycle((
            ActionMap.simple_to_discrete(ActionMap.SIMPLE_ACTIONS.index(FootsiesMove.N_ATTACK))[0],
            ActionMap.simple_to_discrete(ActionMap.SIMPLE_ACTIONS.index(FootsiesMove.STAND))[0]
        )):
            yield action

    def idle():
        while True:
            yield 0

    p1 = idle()

    manager = QLearnerAnalyserManager(
        agent,
        action_dim=ActionMap.n_simple(),
        opponent_action_dim=ActionMap.n_simple(),
    )

    analyser = Analyser(
        env=env,
        # p1_action_source=lambda o, i: next(p1),
        p1_action_source=lambda o, i: agent.act(torch.from_numpy(o).float().unsqueeze(0), i, predicted_opponent_action=manager.get_predicted_opponent_action()),
        custom_elements_callback=manager.add_custom_elements,
        custom_state_update_callback=manager.on_state_update,
    )
    analyser.start(state_change_apply_immediately=True)
