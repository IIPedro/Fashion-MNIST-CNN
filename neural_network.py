from torch import nn


class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.mnist_cnn = nn.Sequential(
            # 28 x 28 x 1
            nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # 14 x 14 x 2
            nn.MaxPool2d(kernel_size=2),
            # 14 x 14 x 2
            nn.Conv2d(2, 4, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            # 14 x 14 x 4 = 392
            nn.Flatten(),
            nn.Linear(784, 784),
            nn.ReLU(),
            # 10
            nn.Linear(784, 10),
        )

    def forward(self, x):
        logits = self.mnist_cnn(x)
        return logits
