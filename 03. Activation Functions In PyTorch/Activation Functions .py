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


# ### Activation Functions :

# 
# Activation functions are mathematical operations applied to the output of a neuron or node in a neural network. They introduce non-linearity to the network, enabling it to model complex relationships and solve a wide range of tasks.

# In[3]:


x = torch.linspace(5, -5, 200)


# In[4]:


x


# In[5]:


x = Variable(x)


# In[6]:


x


# In[7]:


numpy_x = x.data.numpy()


# In[8]:


numpy_x


# ### ReLU ( Rectified Linear Unit ) :

# The ReLU activation function replaces negative input values with zero while leaving positive values unchanged which introduces non-linearity and aids in avoiding vanishing gradient problems during neural network training.

# In[9]:


y_relu = f.relu(x).data.numpy()

plt.plot(numpy_x, y_relu, c = 'blue', label = 'relu')
plt.ylim(-1, 5)
plt.legend(loc = 'best')
plt.show()


# ### Tanh Function :

# Tanh function transforms inputs to values between -1 and 1, introducing non-linearity and symmetry around zero. It's commonly used in neural networks for its bounded output range and suitability for certain architectures like recurrent neural networks (RNNs).

# In[10]:


y_tanh = f.tanh(x).data.numpy()

plt.plot(numpy_x, y_tanh, c = 'blue', label = 'tanh')
plt.ylim(-1.2, 1.2)
plt.legend(loc = 'best')
plt.show()


# ### Sigmoid Function :

# Sigmoid function transforms input values to a range between 0 and 1, suitable for representing probabilities. The curve is S-shaped, with positive inputs mapping to values close to 1, negative inputs to values near 0, and inputs around zero to values near 0.5. It's particularly useful in binary classification tasks but can have vanishing gradient problems for extreme inputs.

# In[11]:


y_sigmoid = f.sigmoid(x).data.numpy()

plt.plot(numpy_x, y_sigmoid, c = 'blue', label = 'sigmoid')
plt.ylim(-0.2, 1.2)
plt.legend(loc = 'best')
plt.show()


# ### Softplus Function :

# 
# Softplus function produces positive values by applying the natural logarithm to the exponential of the input plus one. It introduces non-linearity, resembles the ReLU function for positive inputs, and is often used when outputs don't need to be bounded.

# In[12]:


y_softplus = f.softplus(x).data.numpy()

plt.plot(numpy_x, y_softplus, c = 'blue', label = 'softplus')
plt.ylim(-0.2, 6)
plt.legend(loc = 'best')
plt.show()


# In[ ]:




