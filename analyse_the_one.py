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
from models import the_one_vanilla_no_specials_
from gymnasium.wrappers.transform_observation import TransformObservation


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

    env = TransformObservation(
        FootsiesActionCombinationsDiscretized(
            FlattenObservation(
                FootsiesNormalized(footsies_env)
            )
        ),
        lambda o: torch.from_numpy(o).float().unsqueeze(0),
    )

    agent, _ = the_one_vanilla_no_specials_(
        env.observation_space.shape[0],
        env.action_space.n,
    )

    load_agent_model(agent, "the_one_vanilla_no_specials_tanh")

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
        action_dim=ActionMap.n_simple() - 2,
        opponent_action_dim=ActionMap.n_simple(),
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
        qlearner_manager.add_custom_elements(analyser)
        if mimic_manager is not None:
            mimic_manager.include_mimic_dpg_elements(analyser)

    def custom_state_update_callback(analyser: Analyser):
        qlearner_manager.on_state_update(analyser)
        if mimic_manager is not None:
            mimic_manager.predict_next_move(analyser)

    analyser = Analyser(
        env=env,
        # p1_action_source=lambda o, i: next(p1),
        p1_action_source=agent.act,
        custom_elements_callback=custom_elements_callback,
        custom_state_update_callback=custom_state_update_callback,
    )
    analyser.start(state_change_apply_immediately=True)
