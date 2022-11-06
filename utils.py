import dgl
import torch


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    loop_graphs = [dgl.add_self_loop(graph) for graph in graphs]
    return dgl.batch(loop_graphs), torch.tensor(labels, dtype=torch.long)
