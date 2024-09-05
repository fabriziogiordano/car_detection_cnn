import torch
from model import CarDetectionCNN  # Import your model definition

# Load your quantized model state dictionary
model = CarDetectionCNN()
# Make sure to load without strict to avoid missing keys warnings if using quantization-specific state dicts
model.load_state_dict(
    torch.load(
        "./models/v2/car_detection_cnn.quantized.pth",
        weights_only=True,
        map_location=torch.device("cpu"),
    ),
    strict=False,  # Important for quantized models
)

# Ensure the model is in evaluation mode
model.eval()

# Quantize the model dynamically if not already quantized, since we aim to script the quantized model
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Script the model (converts it into a format suitable for deployment)
try:
    scripted_model = torch.jit.script(model)  # Use torch.jit.script
except Exception as e:
    print("Scripting failed:", e)

# Save the scripted model
scripted_model.save("./models/v2/car_detection_cnn_scripted.quantized.pt")
print("Scripted model saved successfully as car_detection_cnn_scripted.quantized.pt")
