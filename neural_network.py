from torch import nn


class Fashion_MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fashion_mnist_cnn = nn.Sequential(
            # 28 x 28 x 16
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 14 x 14 x 16
            nn.MaxPool2d(kernel_size=2),
            # 14 x 14 x 32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 7 x 7 x 32
            nn.MaxPool2d(kernel_size=2),
            # 7 x 7 x 64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 7 x 7 x 64 = 3136
            nn.Flatten(),
            nn.Linear(3136, 784),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            # 10
            nn.Linear(784, 10),
        )

    def forward(self, x):
        logits = self.fashion_mnist_cnn(x)
        return logits
