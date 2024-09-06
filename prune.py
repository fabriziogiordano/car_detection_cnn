import torch.nn as nn
import torch.nn.utils.prune as prune

def apply_pruning(model):
    """
    Apply pruning to the specified layers of the model.

    Parameters:
    - model (nn.Module): The model to which pruning will be applied.

    Returns:
    - model (nn.Module): The pruned model.
    """
    # Apply pruning to convolutional layers
    prune.random_unstructured(model.conv1, name="weight", amount=0.2)
    prune.random_unstructured(model.conv2, name="weight", amount=0.2)

    # Apply pruning to fully connected layers
    prune.random_unstructured(model.fc1, name="weight", amount=0.2)
    prune.random_unstructured(model.fc2, name="weight", amount=0.2)

    return model

def fine_tune_model(model, train_loader, epochs=10, device='cpu'):
    """
    Fine-tune the pruned model to recover accuracy.

    Parameters:
    - model (nn.Module): The pruned model to be fine-tuned.
    - train_loader (DataLoader): DataLoader containing training data.
    - epochs (int): Number of epochs to fine-tune.
    - device (torch.device): Device to run training on (e.g., 'cuda' or 'cpu').
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

def remove_pruning_hooks(model):
    """
    Remove pruning reparameterization hooks to make pruning permanent.

    Parameters:
    - model (nn.Module): The pruned model with hooks to be removed.
    """
    prune.remove(model.conv1, "weight")
    prune.remove(model.conv2, "weight")
    prune.remove(model.fc1, "weight")
    prune.remove(model.fc2, "weight")
