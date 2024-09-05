import torch
from model import CarDetectionCNN  # Import your model definition

# Load your quantized model state dictionary
model = CarDetectionCNN()
# Make sure to load without strict to avoid missing keys warnings if using quantization-specific state dicts
model.load_state_dict(
    torch.load(
        "parking_lot_model.pth",
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
scripted_model.save("quantized_parking_lot_model_scripted.pt")
print("Scripted model saved successfully as quantized_parking_lot_model_scripted.pt")
