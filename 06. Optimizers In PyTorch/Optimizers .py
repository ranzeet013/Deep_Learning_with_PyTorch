#!/usr/bin/env python
# coding: utf-8

# In[18]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable


# In[2]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
torch.manual_seed(4)


# ### Defining Parameters :

# Learning Rate: Controls how quickly the model adjusts its parameters based on calculated gradients. Higher values lead to faster convergence but could overshoot optimal values. Lower values might lead to slower but more accurate convergence.
# 
# Batch Size: Determines the number of training examples used in each optimization iteration. Smaller sizes add randomness but can converge faster. Larger sizes provide accurate gradient estimates but require more resources.
# 
# Epochs: The number of times the entire dataset is used for training. It ensures the model learns from the data multiple times. Too few epochs can result in underfitting, while too many can lead to overfitting.

# In[3]:


learning_rate = 0.1
batch_size = 32
epochs = 12


# ### Creating Dataset

# Input Data (x): 1000 values evenly spaced between -1 and 1, reshaped to have a shape of (1000, 1).
# 
# Target Data (y): Created by squaring the x values and adding Gaussian noise scaled by 0.1. This simulates a quadratic relationship with added randomness.

# In[4]:


x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim = 1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(* x.size()))

plt.scatter(x.numpy(), y.numpy())
plt.show()


# ### Preparing Dataset : 

# Creating a TensorDataset (torch_dataset) with input x and target y.
# 
#  DataLoader (loader) manages data in batches (size defined by batch_size), shuffles data, and uses parallel loading (num_workers) for efficient training.

# In[8]:


torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset = torch_dataset, 
                         batch_size = batch_size, 
                         shuffle = True, 
                         num_workers = 2)


# ### Defining Neural Network :

# Neural network contains two fully connected layers, hidden and predict. Each layer is represented by a nn.Linear module with an input dimension of 1 and an output dimension of 20. In the forward method, the input undergoes a rectified linear unit (ReLU) activation after passing through the hidden layer. The transformed output is then fed into the predict layer to produce the final output which captures a mapping from a single input to a higher-dimensional hidden space before ultimately predicting an output in the same 20-dimensional space.

# In[28]:


class neuralnet(nn.Module):
    def __init__(self):
        super(neuralnet, self).__init__()
        self.hidden = nn.Linear(1, 20)
        self.predict = nn.Linear(20, 1)
    
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


# In[29]:


net_SGD = neuralnet()
net_Momentum = neuralnet()
net_RMSprop = neuralnet()
net_Adam = neuralnet()
nets = [net_SGD,net_Momentum,net_RMSprop, net_Adam]


# ### Optimizers :

# Optimizer is a key component in training machine learning models, which adjusts model parameters based on calculated gradients 
# to minimize the loss.Popular optimizers include SGD (Stochastic Gradient Descent) and Adam, each with different strategies 
# for updating parameters.Optimizers help models learn from data by iteratively refining their parameters during training.

# In[30]:


opt_SGD  = torch.optim.SGD(net_SGD.parameters(), lr = learning_rate)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr = learning_rate, momentum = 0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(),lr = learning_rate, alpha = 0.9)
opt_Adam  = torch.optim.Adam(net_Adam.parameters(),lr = learning_rate, betas = (0.9,0.99))
optimizers = [opt_SGD, opt_Momentum,opt_RMSprop,opt_Adam]


# ### Creating Loss Function :

# Creating loss function using the mean squared error (MSE) calculation which measures the difference between predictions and
# actual targets.An empty list, losses_his, is prepared to track loss values during training.

# In[34]:


loss_func = torch.nn.MSELoss()
losses_his = [[],[],[],[]]


# ### Training :

# Training for multiple neural network models over a specified number of epochs using various optimizers (SGD, Momentum, RMSprop, and Adam) to update the models' parameter within each epoch, the network's outputs are computed, loss is calculated using mean squared error, and gradients are backpropagated. Loss histories for each optimizer are recorded. Then a plot is generated to visualize how the loss values change during training for each optimizer, aiding in assessing their convergence and performance.

# In[32]:


for epoch in range(epochs):
    print('Epoch:',epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        
        for neuralnet,opt, l_his in zip(nets, optimizers, losses_his):            
            output = neuralnet(b_x)
            loss = loss_func(output, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.data)
            
labels = ['SGD', 'Momentum','RMSprop','Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label = labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()


# In[ ]:




