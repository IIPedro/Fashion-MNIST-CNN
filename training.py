import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from neural_network import Fashion_MNIST_CNN

parser = argparse.ArgumentParser(
    description="A script to train the Fashion MNIST CNN model"
)

# Training hyperparameters
parser.add_argument(
    "-lr",
    "--learning-rate",
    type=float,
    default=3e-4,
    help="Learning rate for the Adam optimizer",
)

parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=64,
    help="Training batch size",
)

parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=5,
    help="Training epochs",
)

args = parser.parse_args()

# Check for NVidia GPU
is_cuda = torch.cuda.is_available()
print(f"GPU available: {is_cuda}")
device = torch.device("cuda" if is_cuda else "cpu")
print(f"Using device: {device}")

# Get training data
training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

# Get test data
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=args.batch_size)
test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

# Load model
model = Fashion_MNIST_CNN()
model.to(device)


def test_loop(dataloader, model, loss_fn, epoch):
    # Test model
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    correct = 0

    # Do not compute gradients
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    writer.add_scalar("Loss/validation", test_loss, epoch)
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def train_loop(dataloader, model, loss_fn, epoch, optimizer):
    # Train model
    model.train()
    size = len(dataloader.dataset)
    loss_sum = 0
    num_batches = 0

    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Accumulate loss for averaging
        loss_sum += loss.item()
        num_batches += 1

        if batch % 100 == 0:
            loss, current = loss.item(), batch * args.batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # Log averaged training loss per epoch
    avg_loss = loss_sum / num_batches
    writer.add_scalar("Loss/train", avg_loss, epoch)


# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/Fashion_MNIST_CNN")

# Enter training loop
for t in range(args.epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, t, optimizer)
    test_loop(test_dataloader, model, loss_fn, t)

print("Done!")

writer.close()

# Save model
torch.save(model.state_dict(), "Fashion_MNIST_CNN.pth")
