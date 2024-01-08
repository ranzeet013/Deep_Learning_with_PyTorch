#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries :

# The torch library is PyTorch's core module that handles tensor operations and computations, serving as the basis for creating neural networks. It offers tools for mathematical operations, GPU acceleration, and automatic differentiation, crucial for efficient model training.

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# ### Preprocessing Data :

# 
# Preprocessing data involves cleaning, scaling, and transforming raw data to make it suitable for machine learning. It includes tasks like handling missing values, scaling features, converting text to numerical representations, and splitting data for training and testing. These steps ensure that the data is in a format that helps models learn effectively and make accurate predictions.

# In[2]:


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# ### Building Neural Net :

# Building a neural network involves designing its structure by selecting the type and number of layers, connecting them, and specifying activation functions. Then, weights and biases are initialized, and forward pass logic is implemented to transform input data into predictions. An appropriate loss function and optimizer are chosen for training. During training, the network learns by adjusting weights using backpropagation and optimization algorithms. Hyperparameters are tuned, and the model's performance is evaluated on validation and test sets. It's an iterative process that culminates in a trained model for making predictions.

# In[3]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()


# ### Optimizer And Loss Function :

# An optimizer and a loss function are essential components in training machine learning models, including neural networks

# ### Optimizer :

# Optimizer is an algorithm that adjusts the parameters (weights and biases) of a model in order to minimize the loss function. The goal is to find the set of parameters that make the model's predictions as accurate as possible. Different optimizers use various strategies to update these parameters based on the gradients of the loss function with respect to the parameters.

# ### Loss Function :

# Loss function quantifies the difference between the predicted values of the model and the actual target values. It measures how well the model is performing and provides a single scalar value that needs to be minimized during training. 

# In[4]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)


# ### Training :

# Training in machine learning is the process of teaching a model to make accurate predictions. It involves feeding labeled data through the model, calculating prediction errors, and adjusting model parameters using optimization techniques to minimize these errors. The model's architecture and parameters are refined through multiple epochs, and its performance is validated and tuned.

# In[6]:


epochs = 20
for epoch in range(epochs):  # epochs is the number of training iterations
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")


# ### Testing :

# Testing in machine learning involves evaluating a trained model's performance on new data that it hasn't seen before. This is done by feeding the data through the model to make predictions and comparing those predictions with the actual values. Metrics like accuracy, precision, recall, and others are calculated to measure the model's performance.

# In[7]:


correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")


# In[ ]:




