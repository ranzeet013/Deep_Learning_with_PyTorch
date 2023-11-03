#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

dataset = pd.read_csv(url, names=column_names)


# In[2]:


dataset


# In[3]:


dataset['species'] = pd.Categorical(dataset['species']).codes

dataset = dataset.sample(frac=1, random_state=1234)


# In[4]:


train_input = dataset.values[:120, :4]
train_target = dataset.values[:120, 4]

test_input = dataset.values[120:, :4]
test_target = dataset.values[120:, 4]


# Feedforward network with one hidden layer containing five units. For activation in the hidden layer, employing the Rectified Linear Unit (ReLU) activation function, defined as f(x) = max(0, x). The output layer consists of three units, where each unit corresponds to one of the three Iris flower classes.

# In[5]:


import torch

torch.manual_seed(1234)

hidden_units = 5

net = torch.nn.Sequential(
    torch.nn.Linear(4, hidden_units),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_units, 3)
)


# Use one-hot encoding for the target data. This means that each class of the flower will be represented as an array (Iris Setosa = [1, 0, 0], Iris Versicolour = [0, 1, 0], and Iris Virginica = [0, 0, 1]), and one element of the array will be the target for one unit of the output layer. When the network classifies a new sample, determine the class by taking the unit with the highest activation value. torch.manual_seed(1234) enables us to use the same random data every time for the reproducibility of results.

# In[6]:


criterion = torch.nn.CrossEntropyLoss()


# Define the loss function as cross-entropy to measure how different the network's output is compared to the target data. Then, set up the stochastic gradient descent (SGD) optimizer with a learning rate of 0.1 and a momentum of 0.9 for training my neural network model.

# In[12]:


optimizer = torch.optim.SGD(net.parameters(), 
                           lr = 0.1, 
                           momentum = 0.9)


# In[13]:


epochs = 50

for epoch in range (epochs):
    inputs = torch.autograd.Variable(torch.Tensor(train_input).float())
    targets = torch.autograd.Variable(torch.Tensor(train_target).long())
    
    optimizer.zero_grad()
    out = net(inputs)
    loss = criterion(out, targets)
    loss.backward()
    optimizer.step()
    
    if epoch == 0 or (epoch + 1) % 10 == 0:
        print('Epoch %d Loss: %.4f' % (epoch + 1, loss.item()))


# In[14]:


import numpy as np

inputs = torch.autograd.Variable(torch.Tensor(test_input).float())
targets = torch.autograd.Variable(torch.Tensor(test_target).long())

optimizer.zero_grad()
out = net(inputs)
_, predicted = torch.max(out.data, 1)

error_count = test_target.size - np.count_nonzero((targets == predicted).numpy())
print('Errors: %d; Accuracy: %d%%' % (error_count, 100 * torch.sum(targets == predicted) / test_target.size))

