import torch.nn as nn
import torch.optim as optim

from evaluate import evaluate


def train(model, train_loader, device):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epoch_losses = []
    for epoch in range(350):
        model.train()
        epoch_loss = 0
        for iter, (batchg, label) in enumerate(train_loader):
            batchg, label = batchg.to(device), label.to(device)
            loss_func = loss_func.to(device)
            prediction = model(batchg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        print(f'epoch: {epoch}, loss {epoch_loss}')
        train_acc = evaluate(model, train_loader, device)
        print(train_acc)
        epoch_losses.append(epoch_loss)
