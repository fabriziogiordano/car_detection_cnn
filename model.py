import torch
import torch.nn as nn
import torch.nn.functional as F


class CarDetectionCNN(nn.Module):
    def __init__(self):
        super(CarDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)  # First convolution layer
        self.conv2 = nn.Conv2d(16, 32, 3, 1)  # Second convolution layer
        self.fc1 = nn.Linear(32 * 30 * 30, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 3)  # Output layer for 3 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 30 * 30)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # Log softmax for classification


class CarDetectionCNN):
    def __init__(self):
        super(CarDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1)  # First convolution layer
        self.conv2 = nn.Conv2d(8, 16, 3, 1)  # Second convolution layer
        self.fc1 = nn.Linear(16 * 30 * 30, 64)  # Fully connected layer
        self.fc2 = nn.Linear(64, 3)  # Output layer for 3 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 16 * 30 * 30)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # Log softmax for classification


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CompactCarDetectionCNN(nn.Module):
    def __init__(self):
        super(CompactCarDetectionCNN, self).__init__()
        self.conv1 = DepthwiseSeparableConv(3, 16, 3)
        self.conv2 = DepthwiseSeparableConv(16, 32, 3)
        self.fc1 = nn.Linear(32 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
