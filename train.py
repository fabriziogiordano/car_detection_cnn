import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import CarDetectionCNN

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data transformations
transform = transforms.Compose([transforms.ToTensor()])

# Load training data and split into train and validation sets
full_data = ImageFolder("data/train", transform=transform)
train_size = int(0.8 * len(full_data))
val_size = len(full_data) - train_size
train_data, val_data = random_split(full_data, [train_size, val_size])

train_loader = DataLoader(
    train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

# Initialize the model, loss function, and optimizer
model = CarDetectionCNN().to(device)  # Move the model to the GPU

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize scaler for mixed precision training
scaler = torch.amp.GradScaler()

# Early stopping parameters
best_val_loss = float("inf")
patience = 10  # Number of epochs to wait for improvement
epochs_no_improve = 0

# Training loop
epochs = 200
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )  # Move data to the GPU
        optimizer.zero_grad()

        # Mixed precision training context
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scale the loss and backward
        scaler.scale(loss).backward()
        # Step optimizer
        scaler.step(optimizer)
        # Update the scale for next iteration
        scaler.update()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}")

    # Check if the validation loss has improved
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(
            model.state_dict(), "./models/v2/best_model.pth"
        )  # Save the best model
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# Load the best model for further usage or inference
model.load_state_dict(torch.load("./models/v2/car_detection_cnn.pth", weights_only=True))

# Quantize the model dynamically for inference
quantized_model = torch.quantization.quantize_dynamic(
    model.to("cpu"),  # Move the model to the CPU before quantization
    {nn.Linear},  # Specify layers to be quantized (e.g., Linear layers)
    dtype=torch.qint8,  # Use 8-bit integer quantization
)

# Save the quantized model state dictionary
torch.save(
    quantized_model.state_dict(), "./models/v2/car_detection_cnn.quantized.pth"
)
