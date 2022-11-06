import numpy as np
import torch
from dgl.data import TUDataset
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from utils import collate
from model.gnn import GIN as GNN
from train import train
from evaluate import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Use {device} for training.')

data_set = TUDataset('MUTAG')
data = np.array(data_set, dtype=object)
labels = np.array([g[1].numpy().tolist() for g in data])

kf = StratifiedKFold(n_splits=10, shuffle=True)

for train_index, test_index in kf.split(data, labels):
    data_train, data_test = data[train_index], data[test_index]
    train_loader = DataLoader(data_train, batch_size=1, collate_fn=collate)
    test_loader = DataLoader(data_test, batch_size=1, collate_fn=collate)
    out_dim = data_set.num_classes
    model = GNN(
        input_dim=1,
        hidden_dim=16,
        output_dim=out_dim,
    )
    train(model, train_loader, device)
    score = evaluate(model, test_loader, device)
    print(score)