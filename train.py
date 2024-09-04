import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import CarDetectionCNN

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load training data
train_data = ImageFolder('data/train', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
model = CarDetectionCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# Save the trained model state dictionary
torch.save(model.state_dict(), 'parking_lot_model.pth')

# Quantize the model dynamically for inference
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {nn.Linear},  # Specify layers to be quantized (e.g., Linear layers)
    dtype=torch.qint8  # Use 8-bit integer quantization
)

# Save the quantized model state dictionary
torch.save(quantized_model.state_dict(), 'quantized_parking_lot_model.pth')
