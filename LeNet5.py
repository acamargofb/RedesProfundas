import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Define the LeNet-5 model
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # Input: 1x28x28, Output: 6x28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0) # Input: 6x28x28, Output: 16x10x10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Fully connected layer, Input: 16x5x5
        self.fc2 = nn.Linear(120, 84)          # Fully connected layer
        self.fc3 = nn.Linear(84, 10)           # Output layer, 10 classes (for MNIST)

    def forward(self, x):
        x = torch.relu(self.conv1(x))          # Conv1 + ReLU
        x = torch.max_pool2d(x, 2)             # MaxPool1 (2x2)
        x = torch.relu(self.conv2(x))          # Conv2 + ReLU
        x = torch.max_pool2d(x, 2)             # MaxPool2 (2x2)
        x = x.view(-1, 16 * 5 * 5)             # Flatten the output from conv layers
        x = torch.relu(self.fc1(x))            # Fully connected layer 1 + ReLU
        x = torch.relu(self.fc2(x))            # Fully connected layer 2 + ReLU
        x = self.fc3(x)                        # Fully connected layer 3 (output)
        return x

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 10

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model, loss function, optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 99:  # Print every 100 mini-batches
            print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

# Testing/Validation function
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f'Accuracy on test images: {100 * correct / total}%')

# Train and validate
for epoch in range(epochs):
    train(epoch)
    test()

# Save the model checkpoint
torch.save(model.state_dict(), 'lenet5_mnist.pth')