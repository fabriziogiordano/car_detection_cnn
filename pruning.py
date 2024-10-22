import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from model import CarDetectionCNN
from prune import apply_pruning, fine_tune_model, remove_pruning_hooks

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([transforms.ToTensor()])

# Load training data
train_data = ImageFolder("data/train", transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Initialize the model
model = CarDetectionCNN()
model.load_state_dict(torch.load("./models/v2/car_detection_cnn.small.pth", map_location=device))
model.to(device)  # Move model to GPU
model.train()  # Set the model to training mode

# Apply pruning
print("Applying pruning...")
model = apply_pruning(model)

# Fine-tune the model
print("Fine-tuning the model...")
fine_tune_model(model, train_loader, epochs=10, device=device)  # Pass device to fine-tune

# Remove pruning hooks
print("Removing pruning hooks...")
remove_pruning_hooks(model)

# Save the pruned model state dictionary
torch.save(model.state_dict(), "./models/v2/car_detection_cnn.small.pruned.pth")

# Quantize the model dynamically for inference
print("Quantizing the model...")
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},  # Specify layers to be quantized (e.g., Linear layers)
    dtype=torch.qint8,  # Use 8-bit integer quantization
)

# Save the quantized model state dictionary
torch.save(quantized_model.state_dict(), "./models/v2/car_detection_cnn_pruned_quantized.pth")
