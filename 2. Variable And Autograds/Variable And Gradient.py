#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.autograd import Variable


# At first I will crear the 2x3 tensor using torch.Tensor

# In[9]:


x_tensor = torch.Tensor(2, 3)
x_tensor


# Then creating the variable from the same tensor created above.

# In[10]:


x_variable = Variable(x_tensor)
x_variable


# ### Variables of Variable :

# In[11]:


x_variable.data


# In[12]:


x_variable = x_variable.grad
print(x_variable)


#  Converted into a variable x_variable with gradient tracking enabled (requires_grad set to True), and printing the output.

# In[22]:


x_tensor = torch.FloatTensor(2, 3)
print(x_tensor.requires_grad)

x_variable = x_tensor.requires_grad_(True)
print(x_variable.requires_grad)


# ### Graph And Gradient :

# In[23]:


x_variable = Variable(x_tensor,volatile=True)
x_variable.grad, x_variable.requires_grad, x_variable.volatile


# I will create a 2x3 tensor x with gradient tracking and calculates the element-wise square and linear terms to form tensor y, and then constructs tensor z with linear transformations of y and returns a tuple of Boolean values indicating gradient tracking for tensors x, y, and z.

# In[20]:


x = Variable(torch.FloatTensor(2, 3),requires_grad=True)
y = x**2 + 4*x
z = 2*y +3

x.requires_grad,y.requires_grad,z.requires_grad


# The .grad attributes of tensors x, y, and z show the calculated gradients, with only x having non-None values.

# In[21]:


gradient = torch.FloatTensor(2, 3)
z.backward(gradient)

print(x.grad)
y.grad,z.grad


# In[ ]:




