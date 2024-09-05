import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def apply_pruning(model):
    # Apply pruning to convolutional layers
    prune.random_unstructured(model.conv1, name="weight", amount=0.2)
    prune.random_unstructured(model.conv2, name="weight", amount=0.2)

    # Apply pruning to fully connected layers
    prune.random_unstructured(model.fc1, name="weight", amount=0.2)
    prune.random_unstructured(model.fc2, name="weight", amount=0.2)

    return model


def fine_tune_model(model, train_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:  # Assuming `train_loader` is defined
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")


def remove_pruning_hooks(model):
    prune.remove(model.conv1, "weight")
    prune.remove(model.conv2, "weight")
    prune.remove(model.fc1, "weight")
    prune.remove(model.fc2, "weight")
