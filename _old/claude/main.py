import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os

# Step 1: Define the dataset
class ParkingLotDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Assume the label is encoded in the filename (e.g., "car_001.jpg" or "empty_001.jpg")
        label = 1 if self.image_files[idx].startswith('car_') else 0
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Step 2: Set up data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 3: Create dataset and dataloader
dataset = ParkingLotDataset('path/to/your/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 4: Define the model (using a pre-trained ResNet18)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Binary classification

# Step 5: Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

# Step 7: Save the trained model
torch.save(model.state_dict(), 'car_detection_model.pth')

# Step 8: Function to use the model for prediction
def predict_image(image_path, model):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return "Car parked" if predicted.item() == 1 else "No car parked"

# Example usage
model.load_state_dict(torch.load('car_detection_model.pth'))
result = predict_image('path/to/test/image.jpg', model)
print(result)