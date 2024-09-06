import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import CarDetectionCNN

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data transformations
transform = transforms.Compose([transforms.ToTensor()])

# Load training data
train_data = ImageFolder("data/train", transform=transform)
# Increase num_workers for faster data loading
train_loader = DataLoader(
    train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)

# Initialize the model, loss function, and optimizer
model = CarDetectionCNN().to(device)  # Move the model to the GPU
# model.load_state_dict(
#     torch.load("./models/v2/car_detection_cnn.pth", map_location=device)
# )
# model.train()  # Set the model to training mode

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize scaler for mixed precision training
scaler = torch.amp.GradScaler()

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
        with torch.amp.autocast(device_type=device):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scale the loss and backward
        scaler.scale(loss).backward()
        # Step optimizer
        scaler.step(optimizer)
        # Update the scale for next iteration
        scaler.update()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# Save the trained model state dictionary
torch.save(model.state_dict(), "./models/v2/car_detection_cnn.small.pth")

# Quantize the model dynamically for inference
quantized_model = torch.quantization.quantize_dynamic(
    model.to("cpu"),  # Move the model to the CPU before quantization
    {nn.Linear},  # Specify layers to be quantized (e.g., Linear layers)
    dtype=torch.qint8,  # Use 8-bit integer quantization
)

# Save the quantized model state dictionary
torch.save(
    quantized_model.state_dict(), "./models/v2/car_detection_cnn.small.quantized.pth"
)
