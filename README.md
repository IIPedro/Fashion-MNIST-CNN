# Fashion-MNIST CNN

This project implements a Convolutional Neural Network (CNN) for classifying images from the Fashion-MNIST dataset. Fashion-MNIST is a dataset consisting of 70,000 grayscale images (28x28 pixels) representing 10 categories of fashion items, such as t-shirts, trousers, dresses, and shoes.

## Project Overview

The CNN model is designed to classify Fashion-MNIST images into their corresponding categories. The architecture consists of three convolutional layers with Batch normalization and ReLU activation, each followed by max pooling to reduce spatial dimensions. The network then flattens the features and passes them through two fully connected layers (3136→784→10) with ReLU activation and dropout regularization (0.3) for the final classification into 10 classes. The model uses CrossEntropy loss and the Adam optimizer for training, with TensorBoard integration for monitoring training and validation metrics.

The project supports both CPU and GPU (CUDA) execution, automatically detecting available NVIDIA hardware.

## File Descriptions

- **`neural_network.py`** — Defines the `Fashion_MNIST_CNN` class, which implements the neural network architecture. The model uses three Conv2d layers (16, 32, and 64 filters with kernel size 3), BatchNorm2d layers, ReLU activation, MaxPool2d layers, a flattened fully connected layer (3136→784) with ReLU dropout (0.3), and a final linear layer producing 10 class logits.

- **`training.py`** — Handles the full training pipeline. It downloads the Fashion-MNIST dataset (if not already present), creates data loaders with batch size 64, configures the optimizer (Adam, learning rate 3e-4), and runs the training loop for 5 epochs. The script supports command-line arguments for learning rate (`-lr`), batch size (`-b`), and epochs (`-e`). Training and validation losses are logged to TensorBoard via `torch.utils.tensorboard`. Validation accuracy is also printed each epoch. The trained model weights are saved to `Fashion_MNIST_CNN.pth`.

- **`inference.py`** — Loads the trained model from `Fashion_MNIST_CNN.pth` and evaluates it on random test samples. It selects 9 random images from the test set, runs them through the model, and saves a 3×3 grid visualization (`predictions.png`) showing the ground truth labels alongside the model predictions, with correct predictions marked with a check mark and incorrect ones with a cross mark.

## Installation Instructions

The project requires the following dependencies, which should be listed in `requirements.txt`:

- `torch` — PyTorch deep learning framework
- `torchvision` — Computer vision utilities and dataset loaders
- `matplotlib` — Plotting library for visualizing predictions
- `numpy` — Numerical computing library
- `tensorboard` — ML visualization library

Install the dependencies using pip:

```
pip install -r requirements.txt
```

### Running the Project

**Train the model:**

```
python training.py
```

You can customize training hyperparameters:

```
python training.py --lr 3e-4 --batch-size 64 --epochs 5
```

Alternatively, download a pretrained model [here](https://huggingface.co/IPedro/Fashion-MNIST-CNN/tree/main).

**Run inference on the trained model:**

```
python inference.py
```

Training metrics can be monitored live using TensorBoard:

```
tensorboard --logdir=runs
```
