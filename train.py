import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import CarDetectionCNNSmall

# Set device to CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Define data transformations
transform = transforms.Compose([transforms.ToTensor()])

# Load training data and split into train and validation sets
train_data = ImageFolder("data/train", transform=transform)
test_data = ImageFolder("data/test", transform=transform)

# Set num_workers to 0 to avoid multiprocessing issues on CPU
train_loader = DataLoader(
    train_data, batch_size=32, shuffle=True, num_workers=0, pin_memory=True
)
test_loader = DataLoader(
    test_data, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
)

# Initialize the model, loss function, and optimizer
model = CarDetectionCNNSmall().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
patience = 50  # Number of epochs to wait for improvement
best_val_loss = float("inf")
epochs_no_improve = 0

# Training loop
epochs = 200
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(test_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

    # Check if the validation loss has improved
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(
            model.state_dict(), "./models/v2/car_detection_cnn.pth"
        )  # Save the best model
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# Load the best model for further usage or inference
model.load_state_dict(torch.load("./models/v2/car_detection_cnn.pth"))

# Quantize the model dynamically for inference
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},  # Specify layers to be quantized (e.g., Linear layers)
    dtype=torch.qint8,  # Use 8-bit integer quantization
)

# Save the quantized model state dictionary
torch.save(quantized_model.state_dict(), "./models/v2/car_detection_cnn_quantized.pth")
