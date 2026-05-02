import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from neural_network import MNIST_CNN

is_cuda = torch.cuda.is_available()
print(f"GPU available: {is_cuda}")
device = torch.device("cuda" if is_cuda else "cpu")

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

test_dataloader = DataLoader(test_data, batch_size=64)

model = MNIST_CNN()
model.to(device)

if os.path.exists("MNIST_CNN.pth"):
    model.load_state_dict(torch.load("MNIST_CNN.pth", map_location=device))
    print("Model loaded!")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


loss_fn = nn.CrossEntropyLoss()

epochs = 1
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
