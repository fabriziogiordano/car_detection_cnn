import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import CarDetectionCNN
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

# Define data transformations
transform = transforms.Compose(
    [
        # Uncomment if resizing is needed
        # transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)


def load_model(model_path, device, quantized=False):
    """
    Load the trained model from the specified path and move it to the specified device.

    Parameters:
    - model_path (str): Path to the model file.
    - device (torch.device): Device to load the model on (CPU or CUDA).
    - quantized (bool): Whether the model to be loaded is quantized.

    Returns:
    - model (nn.Module): Loaded model ready for evaluation.
    """
    model = CarDetectionCNN()

    if quantized:
        # Force device to CPU for quantized models
        device = torch.device("cpu")
        print("Using CPU for quantized model.")
        # Quantize the model dynamically (CPU only)
        model = torch.ao.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

    # Load the model state dict
    state_dict = torch.load(model_path, weights_only=True, map_location=device)

    # Load state dict into the model (use strict=False if quantized)
    model.load_state_dict(state_dict, strict=not quantized)

    # Move model to the specified device
    model.to(device)
    model.eval()

    return model


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test data.

    Parameters:
    - model (nn.Module): The model to be evaluated.
    - test_loader (DataLoader): DataLoader for the test dataset.
    - device (torch.device): Device to perform evaluation on (CPU or CUDA).
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in test_loader:
            if device.type == "cpu":
                images, labels = images.to(device), labels.to(device)
            else:
                images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test set: {accuracy:.2f}%")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model file (.pth)",
    )
    parser.add_argument(
        "--quantized", action="store_true", help="Specify if the model is quantized."
    )

    args = parser.parse_args()

    # Check if CUDA is available and select device accordingly
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.quantized else "cpu"
    )
    print(f"Using device: {device}")

    # Load test data
    test_data = ImageFolder("data/test", transform=transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Load and evaluate the model
    model = load_model(args.model_path, device, quantized=args.quantized)
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
