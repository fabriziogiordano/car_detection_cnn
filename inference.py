import torch
import cv2
from PIL import Image
from torchvision import transforms
from model import SimpleCNN  # Import the model from model.py

# Load the model
model = SimpleCNN()
model.load_state_dict(torch.load('parking_lot_model.pth'))
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Capture image from camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

# Preprocess the image
image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
image = transform(image)
image = image.unsqueeze(0)  # Add batch dimension

# Perform inference
output = model(image)
_, predicted = torch.max(output, 1)

# Print the result
if predicted.item() == 0:
    print("No car detected")
else:
    print("Car detected")
