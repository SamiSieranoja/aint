import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset
import random
import matplotlib.pyplot as plt

# https://github.com/pytorch/examples/blob/main/mnist/main.py

# Optional: Use GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Convert integer labels to one-hot encoding
def to_one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels].to(device)

# Example of one hot encoding:
print(to_one_hot([0,1,2]))
# tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        # [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        # [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])

# Define a simple ReLU-based neural network
class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()
        # self.fc1 = nn.Linear(28 * 28, 64)  # 28 * 28 = 784 pixels
        self.fc1 = nn.Linear(28 * 28, 14)  # 28 * 28 = 784 pixels
        self.fc3 = nn.Linear(14, 10)

    def forward(self, x):
        # Reshape 28x28 image into a vector of size 28 * 28 = 784
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x)) # ReLU 1
        # Scale values to range 0..1 so that the sum = 1
        x = F.softmax(self.fc3(x), dim=1)
        return x

transform = transforms.ToTensor()

fullmnist = datasets.MNIST('.', train=True, download=True, transform=transform)
mnistsubset = fullmnist

# If it runs slowly, you can try using a subset:
# subset_size = 10000
# total_size = len(fullmnist)
# indices = random.sample(range(total_size), subset_size)
# mnistsubset = Subset(fullmnist, indices)

mnist_test = datasets.MNIST('.', train=False, transform=transform)

# Load the MNIST dataset
train_loader = torch.utils.data.DataLoader(
    mnistsubset,
    batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    mnist_test,
    batch_size=1000, shuffle=False
)

# Pick one sample (e.g., index 0)
# image, label = fullmnist[0]

# image is a torch tensor with shape [1, 28, 28] → we need to remove the channel dim for plt
# plt.imshow(image.squeeze(), cmap='gray')
# plt.title(f"Label: {label}")
# plt.axis('off')
# plt.show()


# Initialize model, loss, and optimizer
model = SimpleNeuralNetwork().to(device) # Initialize neural network
criterion = nn.MSELoss() # Loss function

# MSELoss compares the network's prediction to the correct label of the image
# E.g. correct label=5:
#               0   1    2    3    4   5    6   7   8    9
# correct:     [0., 0.,  0.,  0.,  0., 1.,  0., 0., 0.,  0.]])
# network:     [0., 0.0, 0.2, 0.1, 0., 0.5, 0., 0., 0.2, 0.]])
# MSEloss = sum(network - correct))^2

lr = 0.01 # Learning rate
optimizer = optim.Adam(model.parameters(), lr) # Gradient descent based optimization

print("Start training")
# Training loop
epochs = 1 # Iterate through the entire dataset once
for epoch in range(epochs):
    model.train()
    total_loss = 0

   	# Iterate through the dataset in chunks of batch_size=64
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        # print(data.shape)
        target_onehot = to_one_hot(target)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target_onehot)
        loss.backward() # Compute gradient
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
correct = 1
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

accuracy = 100. * correct / len(test_loader.dataset)
print(f"Test Accuracy: {accuracy:.2f}%")

