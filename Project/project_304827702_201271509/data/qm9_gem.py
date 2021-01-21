from schnetpack import SchNet
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import torch
# from torch_geometric.nn import SchNet
from model.nmp_edge import NMPEdge
from model.schnet import SchNet

# DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
dataset = QM9(root='/home/galkampel/tmp/QM9')  # , transform=T.Distance(norm=False)
train_val_set, test_set = torch.utils.data.random_split(dataset, [120000, 9433])
train_set, val_set = torch.utils.data.random_split(train_val_set, [110000, 10000])

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
model = NMPEdge(hidden_channels=256, num_filters=256, hypernet_update=True).to(device)
# model = SchNet(hidden_channels=256, num_filters=256).to(device)


model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
target = 7
n_iter = 1
for i in range(10):
    mae_tot = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.z, batch.pos, batch.batch)
        loss = (pred.view(-1) - batch.y[:, target]).abs().mean()
        loss.backward()
        mae = loss.item()
        mae_tot += mae
        print(f'MAE at iteration {n_iter} = {mae}')
        optimizer.step()
        n_iter += 1
    mae_tot /= len(train_loader)
    print(f'MAE at epoch {i} = {mae_tot}')


# val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
