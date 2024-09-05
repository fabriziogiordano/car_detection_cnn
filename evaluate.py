import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import CarDetectionCNNSmall
import argparse

# Define data transformations
transform = transforms.Compose(
    [
        # transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)


def load_model(model_path):
    """
    Load the trained model from the specified path.

    Parameters:
    - model_path (str): Path to the model file.

    Returns:
    - model (nn.Module): Loaded model ready for evaluation.
    """
    model = CarDetectionCNNSmall()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device("cpu")))
    model.eval()
    return model


def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test dataset.

    Parameters:
    - model (nn.Module): The model to be evaluated.
    - test_loader (DataLoader): DataLoader for the test dataset.

    Prints the accuracy of the model.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
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

    args = parser.parse_args()

    # Load test data
    test_data = ImageFolder("data/test", transform=transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Load and evaluate the model
    model = load_model(args.model_path)
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
