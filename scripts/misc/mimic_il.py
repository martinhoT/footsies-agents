import random
import torch
from data import FootsiesDataset, FootsiesTorchDataset
from torch.utils.data import DataLoader
from agents.action import ActionMap
from tqdm import tqdm
from models import mimic_
from torch.utils.tensorboard import SummaryWriter

summary_writer = SummaryWriter("runs/mimic_il")

dataset = FootsiesDataset.load("footsies-dataset")
dataset = FootsiesTorchDataset(dataset)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

mimic, _ = mimic_(
    observation_space_size=36,
    action_space_size=9,
    dynamic_loss_weights=True,
    dynamic_loss_weights_max=10.0,
    entropy_coef=0.3,
    learning_rate=1e-2,
    scar_size=1,
    scar_min_loss=10.0,
    p1_model=True,
    p2_model=True,
)

# 5% of the dataset
test_states_frac = 0.05
test_states = [(torch.as_tensor(t[0]).float().unsqueeze(0), t[3]) for t in dataset if random.random() < test_states_frac]

step = 0
for epoch in range(10):
    for obs, next_obs, reward, p1_action, p2_action, terminated in tqdm(dataloader):
        p1_action, p2_action = ActionMap.simples_from_transition_torch(obs, next_obs)
        
        if p1_action is not None:
            p1_loss = mimic.p1_model.update(obs, p1_action)
            summary_writer.add_scalar("Learning/Loss of P1's model", p1_loss, step)
            summary_writer.add_scalar("Learning/Scar size of P1's model", mimic.p1_model.number_of_scars, step)
        
        if p2_action is not None:
            p2_loss = mimic.p2_model.update(obs, p2_action)
            summary_writer.add_scalar("Learning/Loss of P2's model", p1_loss, step)
            summary_writer.add_scalar("Learning/Scar size of P2's model", mimic.p2_model.number_of_scars, step)
            
        step += 1

    print("Epoch", epoch, "done")

