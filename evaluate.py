import numpy as np
import torch
from sklearn.metrics import accuracy_score


def evaluate(model, test_loader, device):
    model.eval()
    test_pred, test_label = [], []
    with torch.no_grad():
        for it, (batchg, label) in enumerate(test_loader):
            batchg, label = batchg.to(device), label.to(device)
            pred = np.argmax(model(batchg).cpu(), axis=1).tolist()
            test_pred += pred
            test_label += label.cpu().numpy().tolist()
    return accuracy_score(test_label, test_pred)
