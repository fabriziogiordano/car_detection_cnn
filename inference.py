import torch
from PIL import Image
from torchvision import transforms
from model import CarDetectionCNNSmall

# Load the trained model
model = CarDetectionCNNSmall()
model.load_state_dict(torch.load("./models/v2/car_detection_cnn.small.pth", weights_only=True))
model.eval()

# Define transformations
transform = transforms.Compose(
    [
        # transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)


def classify_image(image_path):
    """
    Classifies an image as 'car parked', 'no car parked', or 'unknown'.
    """
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    classes = ["car parked", "no car parked", "unknown"]
    return classes[predicted.item()]


# Example of classifying an image
image_path = "./data/20240831190201.jpg"
result = classify_image(image_path)
print(f"Result: {result}")
