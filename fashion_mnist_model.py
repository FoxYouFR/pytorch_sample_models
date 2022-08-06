import torch
from torch import nn, optim
from torchvision import datasets, transforms
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Hyperparameters
epochs = 5
lr = 0.005
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 128
n_hidden_3 = 64
n_output = 10

# Model definition
model = nn.Sequential(nn.Linear(n_input, n_hidden_1),
                      nn.ReLU(),
                      nn.Linear(n_hidden_1, n_hidden_2),
                      nn.ReLU(),
                      nn.Linear(n_hidden_2, n_hidden_3),
                      nn.ReLU(),
                      nn.Linear(n_hidden_3, n_output),
                      nn.LogSoftmax(dim=1))

# Loss and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for i in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        
        optimizer.zero_grad()
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss
    else:
        print(f"Training loss at step {i}: {running_loss / len(trainloader)}")
        
dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)

# Calculate probabilities
ps = torch.exp(model(img))

# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')