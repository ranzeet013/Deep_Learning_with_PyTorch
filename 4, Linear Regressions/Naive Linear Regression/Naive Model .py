#!/usr/bin/env python
# coding: utf-8

# In[11]:


import torch
import torch.nn as nn
from torch.autograd import Variable


# In[2]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
torch.manual_seed(3)


# ### Creating Dataset:

#  Creating a dataset from x_train and y_train tensors using PyTorch's TensorDataset The x_train tensor contains input data, while the y_train tensor holds corresponding target values. This approach is commonly used to manage and manipulate data for machine learning tasks, making data handling and processing more organized and efficient.

# In[4]:


x_train = torch.Tensor([[1], [2], [3]])                            #creating x_train tensor
y_train = torch.Tensor([[4], [5], [6]])                            #creating y_train tensir

print('X_train :', x_train)
print('Y_train :', y_train)


# x_train and y_train tensors are converted into Variables. And then visualizing the relationship between the input data and corresponding target values. 

# In[5]:


x, y = Variable(x_train), Variable(y_train)                     #converting x_train, y_train tensor into variable

plt.scatter(x.data.numpy(), y.data.numpy())                     #plotting scatter plot
plt.show()


# In[6]:


x, y


# In[7]:


weight = Variable(torch.rand(1, 1))                               #initializing weight               
weight


# In[8]:


x.mm(weight)                                                     #performing matrix multiplication


# ### Craeting Loss Function:

# Initializing mean squared error (MSE) loss function using nn.MSELoss() module for measuring the difference between predicted and target values.

# In[12]:


loss_func = nn.MSELoss()                                        #initializing loss function
loss_func


# ### Training Naive Linear Regression  :

# Training linear regression model using gradient descent which iterates for 200 steps, adjusting the model's weight to minimize the difference between predicted and target values which involves calculating loss, computing gradients, and updating weights. The scatter plot visualizes the training progress.

# In[21]:


plt.ion()                                                            #interactive mode for real time plorring 

lr = 0.01                                                            #learning rate

for step in range (200):                                             #training in loop
    prediction = x.mm(weight)                                        #predicting using matrix multiplaication
    loss = loss_func(prediction, y)                                  # calculating loss 
    gradient = (prediction - y).view(-1).dot(x.view(-1)) / len(x)    #calculating gradient for weight 
    weight -= lr * gradient                                          #weight using gradient and learning rate
    
    if step % 10 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-')
        plt.title('step %d, cost = %.4f, weight = %.4f,gradient = %.4f' %  (step, loss.data, weight.data[0], gradient.data))
        plt.show()
plt.ioff()


# In[ ]:




