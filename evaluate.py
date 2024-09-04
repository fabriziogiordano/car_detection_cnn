import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import CarDetectionCNN

# Define data transformations
transform = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load test data
test_data = ImageFolder('data/test', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Initialize and load the trained model
model = CarDetectionCNN()
model.load_state_dict(torch.load('parking_lot_model.pth', weights_only=True))
model.eval()

# Evaluation loop
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
