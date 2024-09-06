import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import CarDetectionCNN
import argparse

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
        # Set up the model for quantization
        model.eval()  # Set to evaluation mode before quantization
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        torch.ao.quantization.prepare(model, inplace=True)
        torch.ao.quantization.convert(model, inplace=True)

    # Load the model state dict
    state_dict = torch.load(model_path, map_location=device)

    # Load state dict into the model (use strict=False if quantized)
    model.load_state_dict(state_dict, strict=not quantized)
    model.to(device)  # Move model to the specified device
    model.eval()

    return model


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset.

    Parameters:
    - model (nn.Module): The model to be evaluated.
    - test_loader (DataLoader): DataLoader for the test dataset.
    - device (torch.device): Device to perform evaluation on (CPU or CUDA).

    Prints the accuracy of the model.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(
                device
            )  # Move data to the device
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    test_data = ImageFolder("data/test", transform=transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Load and evaluate the model
    model = load_model(args.model_path, device, quantized=args.quantized)
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
