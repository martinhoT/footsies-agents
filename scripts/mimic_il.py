from torch import nn
from data import FootsiesDataset, FootsiesTorchDataset
from torch.utils.data import DataLoader
from agents.mimic.mimic import PlayerModel, ScarStore, PlayerModelNetwork
from agents.action import ActionMap
from tqdm import tqdm

dataset = FootsiesDataset.load("footsies-dataset")
dataset = FootsiesTorchDataset(dataset)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model = PlayerModel(
    player_model_network=PlayerModelNetwork(
        obs_dim=36,
        action_dim=ActionMap.n_simple(),
        use_sigmoid_output=False,
        input_clip=False,
        input_clip_leaky_coef=0.02,
        hidden_layer_sizes=[64, 64],
        hidden_layer_activation=nn.LeakyReLU,
    ),
    scar_store=ScarStore(
        obs_dim=36,
        max_size=1000,
        min_loss=0.1,
    ),
    learning_rate=1e-2,
    loss_dynamic_weights=True,
)

for obs, next_obs, reward, p1_action, p2_action, terminated in tqdm(dataloader):
    p1_action, _ = ActionMap.simples_from_transition_torch(obs, next_obs)
    if p1_action is not None:
        model.update(obs, p1_action)

