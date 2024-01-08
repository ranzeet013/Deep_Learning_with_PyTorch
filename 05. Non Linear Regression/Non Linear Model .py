#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable


# In[2]:


import numpy
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
torch.manual_seed(4)


# ### Creating Dataset :

# Creating dataset for regression by creating input (x) and output (y) tensors. The input tensor, x, contains values evenly distributed between -1 and 1. The output tensor, y, is created by squaring the x values and adding a small amount of random noise.

# In[3]:


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim = 1)


# In[4]:


x


# In[5]:


y = x.pow(2) + 0.2 * torch.rand(x.size())


# In[6]:


y


# In[7]:


x.shape, y.shape


# convering tensors imtp variables.

# In[8]:


x = Variable(x)


# In[9]:


x


# In[10]:


y = Variable(y)


# In[11]:


y


# plotting the graph from the generated dataframe.

# In[12]:


plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()


# ### Creating Non Linear Model :

# Creating  a non-linear model which has an input layer, a hidden layer with 20 units activated by ReLU, and an output layer.

# In[13]:


hidden = nn.Linear(1, 20, bias=True)

activation = nn.ReLU()

optput = nn.Linear(20, 1, bias=True)

net = nn.Sequential(hidden, activation, optput)


# ### Loss Function :

# Initializing an Adam optimizer for training a neural network with a learning rate of 0.1 with a mean squared error (MSE) loss function (loss_func) for evaluating the difference between predictions and target values which are crucial for training and optimizing neural networks.

# In[15]:


optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()


# ### Training Model :

# Training a neural network model for 200 iterations. In each iteration, it predicts outputs, computes the loss, performs backpropagation, and updates model parameters. Every 10 iterations, it visualizes the data points and predictions, displaying the loss value which allows monitoring of the training progress. After training, interactive mode is turned off.

# In[16]:


for t in range(200):
    prediction = net(x)
    
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if t % 10  == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r--', lw=6)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data, fontdict={'size':20, 'color':'red'})
        plt.show()
        plt.pause(0.2)
        
plt.ioff()


# In[ ]:




