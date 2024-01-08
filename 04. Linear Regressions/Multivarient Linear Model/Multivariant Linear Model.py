#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


# In[2]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
torch.manual_seed(3)


# ### Importing Dataset :

# The dataset consists of records of individual test performances. Each record includes test scores. 

# x_data contains all rows and all columns except the last one, representing the input features. y_data contains all rows and only the last column, representing the target variable. 

# In[3]:


dataframe = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)

x_data = dataframe[:, 0:-1]
y_data = dataframe[:, [-1]]

print('x_data :', x_data)
print('y_data :', y_data)


# x_data and y_data, which are NumPy arrays are converted into PyTorch Variables

# In[4]:


x = Variable(torch.from_numpy(x_data))
y = Variable(torch.from_numpy(y_data))
x, y


# In[5]:


mv_model = nn.Linear(3, 1, bias=True)

print(mv_model)


# In[6]:


print('weigh : ', mv_model.weight)
print('bias  : ', mv_model.bias)


# ### Creating Cost Function :

# Initializing mean squared error (MSE) cost function using nn.MSELoss() module for measuring the difference between predicted and target values.

# In[7]:


cost_func = nn.MSELoss()
cost_func


# In[8]:


optimizer = torch.optim.SGD(mv_model.parameters(), lr=1e-5)


# # Training Multivariant Linear Model :

# Training multivariate linear regression model using a loop that runs for 2000 steps which calculates predictions, computes the cost, performs backpropagation to update model parameters, and monitors progress by printing cost and predictions every 50 steps.

# In[9]:


for step in range(2000):
    optimizer.zero_grad()
    
    prediction = mv_model(x)
    cost = cost_func(prediction, y)
    cost.backward()
    
    optimizer.step()
    
    if step % 50 == 0:
        print(step, "Cost: ", cost.data.numpy(), "\nPrediction:\n", prediction.data.t().numpy())


# In[10]:


accuracy_list = []
for i,real_y in enumerate(y):
    accuracy = (mv_model((x[i])).data.numpy() - real_y.data.numpy())
    accuracy_list.append(np.absolute(accuracy))

for accuracy in accuracy_list:
    print(accuracy)

print("sum accuracy : ",sum(accuracy_list))
print("avg accuracy : ",sum(accuracy_list)/len(y))


# In[ ]:




