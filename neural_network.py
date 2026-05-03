from torch import nn


class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.mnist_cnn = nn.Sequential(
            # 28 x 28 x 16
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # 14 x 14 x 16
            nn.MaxPool2d(kernel_size=2),
            # 14 x 14 x 32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # 7 x 7 x 32
            nn.MaxPool2d(kernel_size=2),
            # 7 x 7 x 64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # 7 x 7 x 64 = 3136
            nn.Flatten(),
            nn.Linear(3136, 3136),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # 10
            nn.Linear(3136, 10),
        )

    def forward(self, x):
        logits = self.mnist_cnn(x)
        return logits
