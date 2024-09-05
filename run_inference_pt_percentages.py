import os

import torch
from PIL import Image
from torchvision import transforms
import argparse

# Define transformations
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)


def load_scripted_model(model_path):
    """
    Load the scripted model from the specified path.

    Parameters:
    - model_path (str): Path to the scripted model (.pt file).

    Returns:
    - model (torch.jit.ScriptModule): Loaded scripted model ready for inference.
    """
    model = torch.jit.load(model_path, map_location=torch.device("cpu"))
    model.eval()  # Set the model to evaluation mode
    return model


def classify_image(model, image_path):
    """
    Classifies an image as 'car parked', 'no car parked', or 'unknown' and returns the probability.

    Parameters:
    - model (torch.jit.ScriptModule): The scripted model for inference.
    - image_path (str): Path to the image to be classified.

    Returns:
    - tuple: (classification result, probability)
    """
    # Load and transform the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(
            output, dim=1
        )  # Apply softmax to get probabilities
        predicted = torch.argmax(
            probabilities, 1
        )  # Get the class with the highest probability

    # Define class labels
    classes = ["car parked", "not car parked", "unknown"]

    # Extract the probability of the predicted class
    predicted_class = predicted.item()
    probability = probabilities[
        0, predicted_class
    ].item()  # Get the probability of the predicted class

    return classes[predicted_class], probability


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Classify an image using a scripted model."
    )
    parser.add_argument(
        "image_path", type=str, help="Path to the image to be classified"
    )
    args = parser.parse_args()

    # Path to the scripted model
    scripted_model_path = "./models/v2/car_detection_cnn_scripted.quantized.pt"
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, scripted_model_path)

    # Load the scripted model
    # print("Loading scripted model...")
    model = load_scripted_model(model_path)

    # Classify the image
    result, probability = classify_image(model, args.image_path)
    print(f"{result}, {probability:.2f}")


if __name__ == "__main__":
    main()
