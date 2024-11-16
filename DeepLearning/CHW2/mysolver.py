import torch
import torch.nn as nn
from torch.utils.data import DataLoader
def train(model, criterion, optimizer, dataloader, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

    return model


def test(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, labels)
            total_loss += loss.item()

    print(f"Test Loss: {total_loss / len(dataloader)}")
