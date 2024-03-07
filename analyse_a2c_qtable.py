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
from main import load_agent_model


AGENT: A2CAgent = None
SIMPLE_ACTION_LABELS = [(move.name, i / ActionMap.n_simple() + (1 / (ActionMap.n_simple() * 2))) for i, move in enumerate(ActionMap.SIMPLE_ACTIONS)]


class QTablePlot:
    def __init__(
        self,
        action_dim: int,
        opponent_action_dim: int,
    ):
        self.action_dim = action_dim
        self.opponent_action_dim = opponent_action_dim
        self.series = None

    def setup(self, title: str = "Q-table"):
        with dpg.group(horizontal=True):
            dpg.add_colormap_scale(min_scale=-1.0, max_scale=1.0, colormap=dpg.mvPlotColormap_Viridis, height=400)
            with dpg.plot(label=title, no_mouse_pos=True, height=400, width=-1) as plot: # height=400, width=-1
                dpg.bind_colormap(plot, dpg.mvPlotColormap_Viridis)
                x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Agent action", lock_min=True, lock_max=True, no_gridlines=True, no_tick_marks=True)
                dpg.set_axis_ticks(x_axis, tuple(SIMPLE_ACTION_LABELS))
                with dpg.plot_axis(dpg.mvYAxis, label="Opponent action", lock_min=True, lock_max=True, no_gridlines=True, no_tick_marks=True) as y_axis:
                    dpg.set_axis_ticks(y_axis, tuple(reversed(SIMPLE_ACTION_LABELS)))
                    initial_data = np.zeros((self.action_dim, self.opponent_action_dim))
                    self.series = dpg.add_heat_series(initial_data.flatten().tolist(), rows=self.opponent_action_dim, cols=self.action_dim, scale_min=-1.0, scale_max=1.0)

    def update(self, q_values: np.ndarray):
        dpg.set_value(self.series, [q_values.flatten().tolist(), [self.opponent_action_dim, self.action_dim]])


class PolicyDistributionPlot:
    def __init__(
        self,
        bar_width: int = 0.2,
    ):
        self.bar_width = bar_width
        
        self.x = np.arange(ActionMap.n_simple())

        # DPG items
        self.x_axis = None
        self.y_axis = None
        self.series = None
    
    def setup(self, title: str = "Policy distribution", width: int = 1050):
        y = np.zeros((ActionMap.n_simple(),))

        with dpg.plot(label=title, width=width) as plot:
            self.x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Move")
            dpg.set_axis_ticks(self.x_axis, tuple([(move.name, i + 1) for i, move in enumerate(ActionMap.SIMPLE_ACTIONS)]))
            dpg.set_axis_limits(self.x_axis, 0.0, ActionMap.n_simple() + 1) # the + 1 is padding

            self.y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Probability")
            dpg.set_axis_limits(self.y_axis, 0.0, 1.0)
        
            self.series = dpg.add_bar_series(self.x + 1.0, y, weight=self.bar_width * 2, parent=self.y_axis)
            dpg.bind_colormap(plot, dpg.mvPlotColormap_Default)

    def update(self, distribution: np.ndarray):
        dpg.set_value(self.series, [list(self.x + 1.0), distribution.tolist()])


class QLearnerAnalyserManager:
    def __init__(
        self,
        action_dim: int,
        opponent_action_dim: int,
    ):
        self.q_table = None
        self.action_dim = action_dim
        self.opponent_action_dim = opponent_action_dim

        self.q_table_plot = QTablePlot(action_dim, opponent_action_dim)
        self.policy_plot = PolicyDistributionPlot()
        
        self.current_observation = None

        # DPG items
        self.predicted_opponent_action = None

    def add_custom_elements(self, analyser: Analyser):
        self.q_table_plot.setup()
        
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
        opponent_action_oh = torch.nn.functional.one_hot(torch.tensor([opponent_action]), num_classes=self.opponent_action_dim).float()
        obs_torch = torch.from_numpy(obs).float().unsqueeze(0)
        obs_torch_with_opp = torch.hstack((obs_torch, opponent_action_oh))
        policy_distribution = AGENT.actor(obs_torch_with_opp).detach().numpy().flatten()
        self.policy_plot.update(policy_distribution)


    def on_state_update(self, analyser: Analyser):
        obs: np.ndarray = analyser.current_observation
        self.current_observation = obs

        q_values = AGENT.critic.q(obs)
        self.q_table_plot.update(q_values)

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

    AGENT = A2CAgent(
        observation_space_size=env.observation_space.shape[0],
        action_space_size=env.action_space.n,
        use_simple_actions=True,
        use_q_table=True,
        consider_opponent_action=True,
        actor_hidden_layer_sizes_specification="64,64",
        actor_hidden_layer_activation_specification="ReLU",
        **{
            "a2c.policy_cumulative_discount": False,
            "critic.discount": 1.0,
            "critic.learning_rate": 0.5,
            "a2c.actor_entropy_loss_coef": 0.1,
            "a2c.actor_optimizer.lr": 0.001,
        }
    )

    load_agent_model(AGENT, "a2c_qlearner")

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
        action_dim=ActionMap.n_simple(),
        opponent_action_dim=ActionMap.n_simple(),
    )

    analyser = Analyser(
        env=env,
        # p1_action_source=mimic_analyser_manager.p2_prediction_discrete,
        p1_action_source=lambda o, i: AGENT.act(torch.from_numpy(o).float().unsqueeze(0), i, predicted_opponent_action=manager.get_predicted_opponent_action()),
        custom_elements_callback=manager.add_custom_elements,
        custom_state_update_callback=manager.on_state_update,
    )
    analyser.start(state_change_apply_immediately=True)
