import torch
from PIL import Image
from torchvision import transforms
from model import SimpleCNN  # Ensure this matches your model definition

# Define transformations
transform = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def load_model(model_path, quantized=False):
    """
    Load the model from the specified path.
    
    Parameters:
    - model_path (str): Path to the model's state dictionary.
    - quantized (bool): Whether to load the model as quantized.
    
    Returns:
    - model (torch.nn.Module): Loaded model ready for inference.
    """
    model = SimpleCNN()
    if quantized:
        # Apply quantization to the model
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

def classify_image(model, image_path):
    """
    Classifies an image as 'car parked', 'no car parked', or 'unknown'.
    
    Parameters:
    - model (torch.nn.Module): The trained model for inference.
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
    classes = ['no car parked', 'car parked', 'unknown']
    return classes[predicted.item()]

def main():
    # Paths to models
    regular_model_path = 'parking_lot_model.pth'
    quantized_model_path = 'quantized_parking_lot_model.pth'

    # Choose whether to use the quantized model
    use_quantized = True  # Set to False to use the regular model

    # Load the appropriate model
    if use_quantized:
        print("Loading quantized model...")
        model = load_model(quantized_model_path, quantized=True)
    else:
        print("Loading regular model...")
        model = load_model(regular_model_path, quantized=False)

    # Specify the path to the image you want to classify
    image_path = './20240831190201.jpg'  # Replace with the actual path of the image

    # Classify the image
    result = classify_image(model, image_path)
    print(f"Result: {result}")

if __name__ == '__main__':
    main()
