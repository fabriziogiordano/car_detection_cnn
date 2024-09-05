import torch
from model import CarDetectionCNN  # Import your model definition

# Load your trained model state dictionary
model = CarDetectionCNN()
model.load_state_dict(
    torch.load(
        "./models/v2/car_detection_cnn.pth",
        weights_only=True,
        map_location=torch.device("cpu"),
    )
)
model.eval()  # Set the model to evaluation mode

# Script the model (converts it into a format suitable for deployment)
scripted_model = torch.jit.script(model)

# Save the scripted model
scripted_model.save("./models/v2/car_detection_cnn_scripted.pt")
