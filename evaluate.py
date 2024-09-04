import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import SimpleCNN  # Import the model from model.py

# Define transformations for the test data
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load the test data
test_data = ImageFolder('data/test', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Load the trained model
model = SimpleCNN()
model.load_state_dict(torch.load('parking_lot_model.pth'))
model.eval()

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
