import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from neural_network import MNIST_CNN

is_cuda = torch.cuda.is_available()
print(f"GPU available: {is_cuda}")
device = torch.device("cuda" if is_cuda else "cpu")

training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)

model = MNIST_CNN()
model.to(device)

learning_rate = 1e-3
batch_size = 64


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
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

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiment_1")

epochs = 15
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    writer.add_scalar("Loss/train", 0.5**t, t)

print("Done!")

writer.close()

torch.save(model.state_dict(), "MNIST_CNN.pth")
