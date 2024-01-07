#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable


# In[2]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
torch.manual_seed(3)


# ### Craeting Dataset :

# Creating a dataset from x_train and y_train tensors using PyTorch's TensorDataset The x_train tensor contains input data, while the y_train tensor holds corresponding target values. This approach is commonly used to manage and manipulate data for machine learning tasks, making data handling and processing more organized and efficient.

# In[4]:


x_train = torch.Tensor([[1], [2], [3]])
y_train = torch.Tensor([[4], [5], [6]])

print("x_train:", x_train)
print("y_train :", y_train)


# x_train and y_train tensors are converted into Variables. And then visualizing the relationship between the input data and corresponding target values.

# In[5]:


x, y = Variable(x_train), Variable(y_train)


# In[6]:


x


# In[7]:


y


# In[9]:


plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()


# In[10]:


weight = Variable(torch.rand(1, 1))


# In[11]:


weight


# In[12]:


x.mm(weight)


# In[13]:


model = nn.Linear(1, 1, bias = True)

print(model)

model.weight, model.bias


# ### Creating Loss Function :

# Initializing mean squared error (MSE) loss function using nn.MSELoss() module for measuring the difference between predicted and target values.

# In[16]:


loss_func = nn.MSELoss()
loss_func


# In[18]:


optimizer = torch.optim.SGD(model.parameters(), 
                            lr = 0.01)


# In[20]:


model(x)


# ### Training Model :

# In a loop of 300 steps, it predicts outputs using the model, calculates the cost (loss), performs backpropagation to compute gradients, updates the model's parameters using an optimizer, and visualizes the training progress. The scatter plot displays data points along with the evolving model's predictions. The process aims to minimize the cost by adjusting the model's weight and bias parameters. After training, interactive mode is turned off.

# In[21]:


plt.ion()

for step in range(300):
    prediction = model(x)
    cost = loss_func(prediction, y)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if step % 10 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'b--')
        plt.title('cost = %.4f, w = %.4f, b = %.4f' % (cost.data,model.weight.data[0][0],model.bias.data))
        plt.show()

plt.ioff()


# In[22]:


x_test = Variable(torch.Tensor([[7]]))
y_test = model(x_test)

print('input : %.4f, output:%.4f' % (x_test.data[0][0], y_test.data[0][0]))


# In[23]:


model.weight, model.bias


# Calculating costs for a range of weights and visualizes them as points on a scatter plot which helps understand how changes in weight affect the cost function.

# In[25]:


W_val, cost_val = [], []

for i in range(-30, 51):
    W = i * 0.1
    model.weight.data.fill_(W)
    cost =  loss_func(model(x),y)
    
    W_val.append(W)
    cost_val.append(cost.data)

plt.plot(W_val, cost_val, 'ro')
plt.show()


# In[ ]:




