from data import FootsiesDataset, FootsiesTorchDataset
from torch.utils.data import DataLoader
from agents.action import ActionMap
from tqdm import tqdm
from models import game_model_
from torch.utils.tensorboard import SummaryWriter
import tyro


def main(
    # game model hyperparameters
    residual: bool = True,
    discrete_conversion: bool = False,
    learning_rate: float = 1e-2,
    # run settings
    run_name: str | None = None,
):
    summary_writer = SummaryWriter("runs/" + run_name) if run_name is not None else None

    dataset = FootsiesDataset.load("footsies-dataset")
    dataset = FootsiesTorchDataset(dataset)

    # We assume we are training sequentially, so no shuffling
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    agent, loggables = game_model_(
        observation_space_size=36,
        action_space_size=9,
        residual=residual,
        discrete_conversion=discrete_conversion,
        discrete_guard=False,
        learning_rate=learning_rate,
        remove_special_moves=False,
    )
    evaluators = loggables["custom_evaluators"]

    step = 0
    for epoch in range(100):
        for obs, next_obs, reward, p1_action, p2_action, terminated in tqdm(dataloader):
            obs = obs.float()
            next_obs = next_obs.float()

            # Discard hitstop/freeze
            if (ActionMap.is_in_hitstop_torch(obs, True) or ActionMap.is_in_hitstop_torch(obs, False)) and obs.isclose(next_obs).all():
                continue

            p1_action, p2_action = ActionMap.simples_from_transition_torch(obs, next_obs)
            
            agent.update_with_simple_actions(obs, p1_action, p2_action, next_obs)

            if summary_writer is not None:
                for (tag, evaluator) in evaluators:
                    summary_writer.add_scalar(tag, evaluator(), step)

            step += 1

        print("Epoch", epoch, "done")

if __name__ == "__main__":
    tyro.cli(main)