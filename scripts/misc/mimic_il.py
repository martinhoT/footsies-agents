import random
import torch
from data import FootsiesDataset, FootsiesTorchDataset
from torch.utils.data import DataLoader
from agents.action import ActionMap
from tqdm import tqdm
from models import mimic_
from torch.utils.tensorboard import SummaryWriter
from agents.torch_utils import ActionHistoryAugmentation, TimeSinceLastCommitAugmentation
import tyro


def main(
    # mimic hyperparameters
    dynamic_loss_weights: bool = True,
    entropy_coef: float = 0.3,
    scar_size: int = 1,
    scar_min_loss: float = 10000.0,
    recurrent: bool = False,
    # wrappers
    history_p1: bool = False,
    history_p2: bool = False,
    time_p1: bool = False,
    time_p2: bool = False,
    history_p1_size: int = 5,
    history_p2_size: int = 5,
    history_p1_distinct: bool = True,
    history_p2_distinct: bool = True,
    # run settings
    run_name: str | None = None,
):
    if run_name is not None:
        summary_writer = SummaryWriter("runs/" + run_name)

    append_action_augment_p1 = ActionHistoryAugmentation(history_p1_size, 9, history_p1_distinct)
    append_action_augment_p2 = ActionHistoryAugmentation(history_p2_size, 9, history_p2_distinct)
    time_augment_p1 = TimeSinceLastCommitAugmentation(120)
    time_augment_p2 = TimeSinceLastCommitAugmentation(120)

    append_action_augment_p1.enabled = history_p1
    append_action_augment_p2.enabled = history_p2
    time_augment_p1.enabled = time_p1
    time_augment_p2.enabled = time_p2

    dataset = FootsiesDataset.load("footsies-dataset")
    dataset = FootsiesTorchDataset(dataset)

    # We assume we are training sequentially, so no shuffling
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    mimic, _ = mimic_(
        observation_space_size=36
            + append_action_augment_p1.enabled * append_action_augment_p1.action_dim * append_action_augment_p1.history.maxlen
            + append_action_augment_p2.enabled * append_action_augment_p2.action_dim * append_action_augment_p2.history.maxlen
            + time_augment_p1.enabled * 1
            + time_augment_p2.enabled * 1,
        action_space_size=9,
        dynamic_loss_weights=dynamic_loss_weights,
        dynamic_loss_weights_max=10.0,
        entropy_coef=entropy_coef,
        learning_rate=1e-2,
        scar_size=scar_size,
        scar_min_loss=scar_min_loss,
        p1_model=True,
        p2_model=True,
        recurrent=recurrent,
    )

    # 5% of the dataset
    test_states_frac = 0.05
    test_states = [(torch.as_tensor(t[0]).float().unsqueeze(0), t[3]) for t in dataset if random.random() < test_states_frac]

    step = 0
    p1_action_prev = 0
    p2_action_prev = 0
    for epoch in range(100):
        for obs, next_obs, reward, p1_action, p2_action, terminated in tqdm(dataloader):
            
            p1_action, p2_action = ActionMap.simples_from_transition_torch(obs, next_obs)

            if append_action_augment_p1.enabled:
                obs = append_action_augment_p1(obs, p1_action_prev)
            if append_action_augment_p2.enabled:
                obs = append_action_augment_p2(obs, p2_action_prev)
            if time_augment_p1.enabled:
                obs = time_augment_p1(obs, p1_action_prev)
            if time_augment_p2.enabled:
                obs = time_augment_p2(obs, p2_action_prev)
            
            if p1_action is not None:
                p1_loss = mimic.p1_model.update(obs, p1_action, terminated)
                if p1_loss is not None and run_name is not None:
                    summary_writer.add_scalar("Learning/Loss of P1's model", p1_loss, step)
                    summary_writer.add_scalar("Learning/Scar size of P1's model", mimic.p1_model.number_of_scars, step)
            
            if p2_action is not None:
                p2_loss = mimic.p2_model.update(obs, p2_action, terminated)
                if p2_loss is not None and run_name is not None:
                    summary_writer.add_scalar("Learning/Loss of P2's model", p2_loss, step)
                    summary_writer.add_scalar("Learning/Scar size of P2's model", mimic.p2_model.number_of_scars, step)
                
            step += 1

            p1_action_prev = p1_action
            p2_action_prev = p2_action

            if terminated:
                append_action_augment_p1.reset()
                append_action_augment_p2.reset()
                time_augment_p1.reset()
                time_augment_p2.reset()

        print("Epoch", epoch, "done")


if __name__ == "__main__":
    tyro.cli(main)