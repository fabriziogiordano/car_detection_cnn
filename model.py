import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)  # First convolution layer
        self.conv2 = nn.Conv2d(16, 32, 3, 1) # Second convolution layer
        self.fc1 = nn.Linear(32 * 30 * 30, 128) # Fully connected layer
        self.fc2 = nn.Linear(128, 3)        # Output layer for 3 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 30 * 30)        # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)      # Log softmax for classification
