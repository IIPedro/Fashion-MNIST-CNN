from torch import nn


class Fashion_MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1: 1 -> 16 -> 32
        self.conv1_1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.skip1 = nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0, bias=True)

        # Block 2: 32 -> 64 -> 64
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.skip2 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=True)

        # Block 3: 64 -> 32 -> 16
        self.conv3_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3_1 = nn.BatchNorm2d(32)
        self.conv3_2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3_2 = nn.BatchNorm2d(16)
        self.skip3 = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=True)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3136, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Block 1
        identity = self.skip1(x)
        out = self.relu(self.bn1_1(self.conv1_1(x)))
        out = self.bn1_2(self.conv1_2(out))
        out = self.relu(out + identity)
        out = self.pool(out)

        # Block 2
        identity = self.skip2(out)
        out = self.relu(self.bn2_1(self.conv2_1(out)))
        out = self.bn2_2(self.conv2_2(out))
        out = self.relu(out + identity)

        # Block 3
        identity = self.skip3(out)
        out = self.relu(self.bn3_1(self.conv3_1(out)))
        out = self.bn3_2(self.conv3_2(out))
        out = self.relu(out + identity)

        # Classifier
        out = self.flatten(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits
