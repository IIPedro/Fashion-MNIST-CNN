from torch import nn


class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.mnist_cnn = nn.Sequential(
            # 28 x 28 x 16
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # 28 x 28 x 32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # 14 x 14 x 32
            nn.MaxPool2d(kernel_size=2),
            # 14 x 14 x 32
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # 14 x 14 x 32
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # 14 x 14 x 32 = 6272
            nn.Flatten(),
            nn.Linear(6272, 6272),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # 6272
            nn.Linear(6272, 6272),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # 10
            nn.Linear(6272, 10),
        )

    def forward(self, x):
        logits = self.mnist_cnn(x)
        return logits
