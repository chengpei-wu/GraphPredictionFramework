import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv

from model.cnn import CNN
from model.mlp import MLP


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        for layer in range(num_layers - 1):
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.cnn = CNN(output_dim)

    def forward(self, g):
        h = g.in_degrees().view(-1, 1).float()
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        h = torch.unsqueeze(h, dim=0)
        h = self.cnn(h)
        # print(h)
        return h
