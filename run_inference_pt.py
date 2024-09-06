import torch
from PIL import Image
from torchvision import transforms

# Define transformations
transform = transforms.Compose(
    [
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
    Classifies an image as 'car parked', 'no car parked', or 'unknown'.

    Parameters:
    - model (torch.jit.ScriptModule): The scripted model for inference.
    - image_path (str): Path to the image to be classified.

    Returns:
    - str: Classification result ('car parked', 'no car parked', 'unknown').
    """
    # Load and transform the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    # Define class labels
    classes = ["car parked", "not car parked", "unknown"]
    return classes[predicted.item()]


def main():
    # Path to the scripted model
    #scripted_model_path = "car_detection_cnn_scripted.pt"
    scripted_model_path = "./models/v2/car_detection_cnn_scripted_quantized.pt"

    # Load the scripted model
    print("Loading scripted model...")
    model = load_scripted_model(scripted_model_path)

    # Specify the path to the image you want to classify
    image_path = "./data/20240831190201.jpg"  # Car Parked
    #image_path = "./20240902082401.jpg"  # Car Not Parked

    # Classify the image
    result = classify_image(model, image_path)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
