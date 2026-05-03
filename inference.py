import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from neural_network import Fashion_MNIST_CNN

# Check for NVidia GPU
is_cuda = torch.cuda.is_available()
print(f"GPU available: {is_cuda}")
device = torch.device("cuda" if is_cuda else "cpu")
print(f"Using device: {device}")

# Get test data
transform = transforms.Compose([transforms.ToTensor()])
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=transform
)

# Fashion-MNIST class labels
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Load model from disk
model = Fashion_MNIST_CNN()
_ = model.load_state_dict(
    torch.load("Fashion_MNIST_CNN.pth", map_location=device, weights_only=True)
)
model = model.to(device)
model.eval()
print("Model loaded!")


# Helper: denormalise an image back to [0,1]
def im_convert(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor (C,H,W) to numpy array (H,W,C) in [0,1]."""
    image = tensor.cpu().numpy().squeeze()
    return np.clip(image, 0, 1)


# Pick 9 random samples and plot a 3×3 grid
indices = np.random.choice(len(test_data), 9, replace=False).tolist()
images, labels = torch.utils.data.dataloader.default_collate(
    [test_data[i] for i in indices]
)
images = images.to(device)
labels = labels.to(device)

# Make predictions
with torch.no_grad():
    logits = model(images)
    preds = logits.argmax(dim=1)

# Plot 3×3 grid
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for i, ax in enumerate(fig.axes):
    if i < 9:
        ax.imshow(im_convert(images[i]), cmap="gray")
        exp_name = CLASS_NAMES[labels[i].item()]
        pred_name = CLASS_NAMES[preds[i].item()]
        correct = "✓" if labels[i].item() == preds[i].item() else "✗"
        ax.set_title(f"Exp: {exp_name}\nPred: {pred_name} {correct}")
        ax.axis("off")

plt.tight_layout()
plt.savefig("predictions.png", dpi=150)
plt.show()
print("Predictions saved to predictions.png")
