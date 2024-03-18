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
from main import load_agent_model
from analyse_a2c_qlearner import QLearnerAnalyserManager
from analyse_opponent_model import MimicAnalyserManager
from models import to_no_specials_, to_no_specials_opp_, to_
from gymnasium.wrappers.transform_observation import TransformObservation
from main import setup_logger
from opponents.curriculum import WhiffPunisher, Backer, UnsafePunisher
from agents.utils import FootsiesPhasicMoveProgress
import logging


if __name__ == "__main__":

    setup_logger("analyse", stdout_level=logging.DEBUG, log_to_file=False)

    custom_opponent = Backer()

    footsies_env = FootsiesEnv(
        game_path="../Footsies-Gym/Build/FOOTSIES.x86_64",
        game_port=15000,
        opponent_port=15001,
        remote_control_port=15002,
        render_mode="human",
        sync_mode="synced_non_blocking",
        fast_forward=False,
        dense_reward=True,
        # vs_player=True,
        opponent=custom_opponent.act,
    )

    env = TransformObservation(
        FootsiesActionCombinationsDiscretized(
            FlattenObservation(
                # FootsiesPhasicMoveProgress(
                    FootsiesNormalized(
                        footsies_env,
                        # normalize_guard=False,
                    )
                # )
            )
        ),
        lambda o: torch.from_numpy(o).float().unsqueeze(0),
    )

    agent, loggables = to_no_specials_(
        env.observation_space.shape[0],
        env.action_space.n,
        rollback=True,
        actor_entropy_coef=0.5,
        critic_agent_update="q_learning",
        critic_opponent_update="q_learning",
        # critic_target_update_rate=1000,
        critic_table=True,
    )

    # idle_distribution = torch.tensor([0.0] * ActionMap.n_simple()).float().unsqueeze(0)
    # idle_distribution[0, 0] = 1.0

    load_agent_model(agent, "curriculum_greedies_table")

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

    qlearner_manager = QLearnerAnalyserManager(
        agent.a2c,
        action_dim=agent.action_dim,
        opponent_action_dim=agent.opponent_action_dim,
    )

    if agent.opponent_model is not None:
        mimic_manager = MimicAnalyserManager(
            p1_model=None,
            p2_model=agent.opponent_model,
            p1_mirror_p2=False,
        )
    
    else:
        mimic_manager = None

    def custom_elements_callback(analyser: Analyser):
        with dpg.group(horizontal=True):
            dpg.add_text("Predicted opponent action:")
            dpg.add_text("X", tag="predicted_opponent_action")

        qlearner_manager.add_custom_elements(analyser)
        if mimic_manager is not None:
            mimic_manager.include_mimic_dpg_elements(analyser)

    def custom_state_update_callback(analyser: Analyser):
        qlearner_manager.on_state_update(analyser)
        if mimic_manager is not None:
            mimic_manager.predict_next_move(analyser)

    def act(obs, info):
        # This is so bad, it's a repeat of update()'s last part (since the the_one doesn't have update() called, only the A2CAgent), but idc
        prev_obs = agent.a2c.current_info
        if prev_obs is not None:
            _, opponent_action = ActionMap.simples_from_transition_ori(prev_obs, info)
            agent.previous_valid_opponent_action = opponent_action if opponent_action is not None else agent.previous_valid_opponent_action

        action = agent.act(obs, info)

        if agent.recently_predicted_opponent_action is not None:
            predicted_opponent_action = ActionMap.simple_as_move(agent.recently_predicted_opponent_action)
        else:
            predicted_opponent_action = None
        dpg.set_value("predicted_opponent_action", predicted_opponent_action.name if predicted_opponent_action is not None else "X")

        return action

    analyser = Analyser(
        env=env,
        # p1_action_source=lambda o, i: next(p1),
        p1_action_source=act,
        custom_elements_callback=custom_elements_callback,
        custom_state_update_callback=custom_state_update_callback,
    )
    analyser.start(state_change_apply_immediately=True)
