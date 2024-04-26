import torch
from models import mimic_
from torch.utils.data import DataLoader
from typing import Callable, Iterator
from agents.mimic.agent import MimicAgent


def test_non_recurrent(il_dataset: Iterator):
    mimic, _ = mimic_(
        observation_space_size=36,
        action_space_size=9,
        dynamic_loss_weights=True,
        dynamic_loss_weights_max=10.0,
        entropy_coef=0.3,
        learning_rate=1e-2,
        scar_size=1,
        scar_min_loss=float("+inf"),
        use_p1_model=True,
        use_p2_model=True,
        recurrent=False,
    )

    assert mimic.p1_model is not None
    assert mimic.p2_model is not None
    assert mimic.learn_p1
    assert mimic.learn_p2
    assert not mimic.p1_model.network.is_recurrent
    assert not mimic.p2_model.network.is_recurrent

    for obs, _, _, p1_action, p2_action, terminated in il_dataset():
        mimic.update_with_simple_actions(obs, p1_action, p2_action, terminated)
        
        # Only train for one episode, since here it takes longer
        if terminated:
            break
   
    p1_loss = mimic.evaluate_p1_average_loss_and_clear()
    p2_loss = mimic.evaluate_p2_average_loss_and_clear()

    # TODO: loss values not checked
    assert 0.0 < p1_loss < 5.0
    assert 0.0 < p2_loss < 5.0


def recurrent_run(mimic: MimicAgent, il_dataset: Callable[[], Iterator]) -> tuple[float, float]:
    assert mimic.p1_model is not None
    assert mimic.p2_model is not None
    assert mimic.learn_p1
    assert mimic.learn_p2
    assert mimic.p1_model.network.is_recurrent
    assert mimic.p2_model.network.is_recurrent
    
    # Make sure the internal hidden state is properly managed on update
    p1_hidden = None
    p2_hidden = None
    for obs, _, _, p1_action, p2_action, terminated in il_dataset():
        if p1_hidden is None:
            assert mimic.p1_model.network.hidden is None
        else:
            assert torch.isclose(p1_hidden, mimic.p1_model.network.hidden).all()
        
        if p2_hidden is None:
            assert mimic.p2_model.network.hidden is None
        else:
            assert torch.isclose(p2_hidden, mimic.p2_model.network.hidden).all()

        if p1_action is not None:
            _, p1_hidden = mimic.p1_model.network._recurrent(obs, p1_hidden)
            p1_hidden = p1_hidden.detach()
        if p2_action is not None:
            _, p2_hidden = mimic.p2_model.network._recurrent(obs, p2_hidden)
            p2_hidden = p2_hidden.detach()

        mimic.update_with_simple_actions(obs, p1_action, p2_action, terminated)

        if terminated:
            p1_hidden = None
            p2_hidden = None
    
    p1_loss = mimic.evaluate_p1_average_loss_and_clear()
    p2_loss = mimic.evaluate_p2_average_loss_and_clear()

    return p1_loss, p2_loss


def test_recurrent(il_dataset: Callable[[], Iterator]):
    mimic, _ = mimic_(
        observation_space_size=36,
        action_space_size=9,
        dynamic_loss_weights=False,
        dynamic_loss_weights_max=10.0,
        entropy_coef=0.3,
        learning_rate=1e-2,
        scar_size=1,
        scar_min_loss=float("+inf"),
        use_p1_model=True,
        use_p2_model=True,
        recurrent=True,
    )
    
    p1_loss, p2_loss = recurrent_run(mimic, il_dataset)

    assert 0.435530 < p1_loss < 0.435531
    assert 0.401413 < p2_loss < 0.401414


def test_recurrent_dynamic_loss(il_dataset: Callable[[], Iterator]):
    mimic, _ = mimic_(
        observation_space_size=36,
        action_space_size=9,
        dynamic_loss_weights=True,
        dynamic_loss_weights_max=10.0,
        entropy_coef=0.3,
        learning_rate=1e-2,
        scar_size=1,
        scar_min_loss=float("+inf"),
        use_p1_model=True,
        use_p2_model=True,
        recurrent=True,
    )
    
    p1_loss, p2_loss = recurrent_run(mimic, il_dataset)

    assert 2.086574 < p1_loss < 2.086575
    assert 1.757696 < p2_loss < 1.757697