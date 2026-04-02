import torch
from torch import nn
# vediamo se ora funzionaaaaa

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=2) # from 3x64x64 to 8x32x32
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2) # from 8x32x32 to 16x16x16
        self.bn2 = nn.BatchNorm2d(16)
        # Add more layers...

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=2, stride=2) # from 16x16x16 to 16x8x8
        self.bn3 = nn.BatchNorm2d(16)

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass

        # B x 3 x 64 x 64
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return x