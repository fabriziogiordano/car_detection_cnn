import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(3, 16, 3, 1)  # 3 input channels (RGB), 16 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(16, 32, 3, 1)  # 16 input channels, 32 output channels, 3x3 kernel
        self.fc1 = nn.Linear(32 * 30 * 30, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 2)  # Output layer (2 classes: car, no_car)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 30 * 30)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
