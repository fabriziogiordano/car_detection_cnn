import os
import sys
import warnings
import torch
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify
from io import BytesIO

app = Flask(__name__)

# Suppress the warning
warnings.filterwarnings("ignore", category=UserWarning, module='flask')


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
    # If running in a PyInstaller bundle, use _MEIPASS to find the file
    if getattr(sys, "frozen", False):
        model_path = os.path.join(sys._MEIPASS, model_path)

    model = torch.jit.load(model_path, map_location=torch.device("cpu"))
    model.eval()  # Set the model to evaluation mode
    return model


def classify_image(model, image):
    """
    Classifies an image as 'car parked', 'no car parked', or 'unknown' and returns the probability.

    Parameters:
    - model (torch.jit.ScriptModule): The scripted model for inference.
    - image (PIL.Image): Image to be classified.

    Returns:
    - tuple: (classification result, probability)
    """
    # Transform the image
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


# Load the model once when the server starts
model_file = "models/prod/car_detection_cnn_scripted.pt"
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, model_file)
model = load_scripted_model(model_path)


@app.route("/classify", methods=["POST"])
def classify():
    """
    Endpoint to classify an image. Expects an image file in the request.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(file.stream)
        result, probability = classify_image(model, image)
        return jsonify({"result": result, "probability": probability})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
