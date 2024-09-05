To deploy your model on the Raspberry Pi without needing the entire codebase, you can bundle the model into a self-contained format that includes everything required for inference. This can be achieved by:

1. **Script Module (`torch.jit.script`)**: Converts your model into a TorchScript format, which encapsulates the model's architecture and weights in a single file. This approach is recommended for deployment since it allows you to run the model independently of the original Python code.

2. **Exporting the Model with TorchScript**:
   - You can use `torch.jit.script` or `torch.jit.trace` to create a serialized model that can be loaded and run without needing the full Python model definition.

Here's how to do it:

### Step-by-Step: Exporting and Using a TorchScript Model

1. **Script the Model**: Convert your trained model to TorchScript format and save it.

   ```python
   import torch
   from model import CarDetectionCNN  # Import your model definition

   # Load your trained model state dictionary
   model = CarDetectionCNN()
   model.load_state_dict(torch.load('./models/car_detection_cnn.pth', map_location=torch.device('cpu')))
   model.eval()  # Set the model to evaluation mode

   # Script the model (converts it into a format suitable for deployment)
   scripted_model = torch.jit.script(model)

   # Save the scripted model
   scripted_model.save('./models/car_detection_cnn_scripted.pt')
   ```

   - This step creates a `.pt` file (`./models/car_detection_cnn_scripted.pt`) that contains the entire model in a serialized form.

2. **Using the Scripted Model on Raspberry Pi**: Load and run inference using the scripted model without requiring the full model code.

   Here's an updated `run_inference.py` script for using the scripted model:

   ```python
   import torch
   from PIL import Image
   from torchvision import transforms

   # Define transformations
   transform = transforms.Compose([
       transforms.Resize((128, 128)),  # Ensure this matches your training
       transforms.ToTensor(),
   ])

   def load_scripted_model(model_path):
       """
       Load the scripted model from the specified path.
       
       Parameters:
       - model_path (str): Path to the scripted model (.pt file).
       
       Returns:
       - model (torch.jit.ScriptModule): Loaded scripted model ready for inference.
       """
       model = torch.jit.load(model_path, map_location=torch.device('cpu'))
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
       classes = ['no car parked', 'car parked', 'unknown']
       return classes[predicted.item()]

   def main():
       # Path to the scripted model
       scripted_model_path = './models/car_detection_cnn_scripted.pt'

       # Load the scripted model
       print("Loading scripted model...")
       model = load_scripted_model(scripted_model_path)

       # Specify the path to the image you want to classify
       image_path = '/path/to/new_image.jpg'  # Replace with the actual path of the image

       # Classify the image
       result = classify_image(model, image_path)
       print(f"Result: {result}")

   if __name__ == '__main__':
       main()
   ```

### Advantages of Using TorchScript:
- **Self-contained**: The `.pt` file includes both the model structure and weights, eliminating the need for the original model definition code.
- **Efficiency**: TorchScript models are optimized for performance and can run without the full Python runtime, reducing deployment complexity.
- **Portability**: The serialized model can be easily transferred to other devices (like a Raspberry Pi) and used for inference.

### Deployment on Raspberry Pi:
- Transfer the `car_detection_cnn_scripted.pt` file to your Raspberry Pi.
- Run the `run_inference.py` script on the Pi using:

  ```bash
  python3 run_inference.py
  ```

By using this approach, you simplify the deployment process and ensure that all necessary components are bundled within the serialized model file.