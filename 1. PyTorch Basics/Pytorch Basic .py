#!/usr/bin/env python
# coding: utf-8

# ## Pytorch :

# PyTorch is an open-source machine learning framework that excels in building and training deep learning models. Its dynamic computation graph, automatic differentiation, and GPU support make it popular among researchers and developers. It provides tools for creating tensors, defining neural networks, loading data, and optimizing models.

# In[1]:


import torch


# ### Tensors :

#  Tensors are multi-dimensional arrays similar to NumPy arrays but with additional capabilities optimized for deep learning. Tensors in PyTorch are the fundamental building blocks for creating and manipulating data in the framework.

# ### Tensor Basic :

# 'torch.tensor' is used for creating and manipulating multi-dimensional arrays (tensors) used in deep learning.

# In[3]:


x = torch.Tensor(2, 3)
print(x)


# In[4]:


print("Tensor size :", x.size())
print("Tensor type :", x.type())


# ### Tensor With Characterstics :

# torch.zeros creates a tensor filled with zeros of a specified size.

# In[5]:


x = torch.zeros(2, 3)
print(x)


# In[6]:


print("Tensor size:", x.size())
print("Temsor type:", x.type())


# In[9]:


x = torch.zeros(2,3, dtype=torch.int32)
print(x)


# In[10]:


print("Torch size:", x.size())
print("Torch typr:", x.type())


# torch.ones generates a tensor filled with ones of a specified size.

# In[11]:


x = torch.ones(2, 3)
print(x)


# In[12]:


print("Torch size:", x.size())
print("Torch type:", x.type())


# 
# torch.rand produces a tensor with random values between 0 and 1 of a specified size.

# In[13]:


x = torch.rand(2, 3)
print(x)


# In[14]:


print("Tensor type:", x.type())
print("Tensor size", x.size())


# torch.randn generates a tensor with random values drawn from a standard normal distribution (mean 0, standard deviation 1) of a specified size.

# In[15]:


x = torch.randn(2, 3)
print(x)
print('Tensor size', x.size())
print('Tensor type:', x.type())


# torch.eye creates an identity matrix as a tensor with ones on the diagonal and zeros elsewhere, of a specified size.

# In[17]:


x = torch.eye(2, 3)
print(x)
print("Tensor size:", x.size())
print("Tensor type:", x.type())


# 
# torch.arange generates a tensor with a sequence of values from start to end with a specified step size.

# In[19]:


x = torch.arange(2, 3)
print(x)
print("Tensor size:", x.size())
print("Tensor type:", x.type())


# torch.tensor is a constructor that creates a new PyTorch tensor from a given data source, like a list or an array-like structure.

# In[20]:


x = torch.Tensor(2, 3)
print(x)
print("Tensor size:", x.size())
print("Tensor type:", x.type())


# Create a tensor of data type float

# In[23]:


x = torch.FloatTensor(2, 3)
print(x)
print("Tensor size:", x.size())
print("Tensor type:", x.type())


# Creates a tensor of data type int

# In[24]:


x = torch.IntTensor(2,3)
print(x)
print("Tensor Type : ", x.type())
print("Tensor Size : ", x.size())


# ### Tensor And NUmpy :

# NumPy is a Python library for numerical computations, providing support for multi-dimensional arrays (tensors), mathematical functions, and operations, making it essential for scientific computing and data analysis.

# In[25]:


import numpy as np


# In[29]:


x = np.array([[1, 2, 3], [1, 2, 3]])
print(x, type(x))


# torch.from_numpy creates a PyTorch tensor from a NumPy array 

# In[30]:


x = torch.from_numpy(x)
print(x)
print("Tensor size :", x.size())
print("Tensor type :", x.type())


# ### Tensor Slicing :

# Tensor slicing involves extracting a portion of a tensor using indexing. It's a way to access specific elements, rows, columns, or sub-tensors.

# In[35]:


x = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(x)
print('Tensor size:', x.size())
print('Tensor type:', x.type())


# Slice all dimensions of a tensor x in one line, effectively selecting all elements.

# In[36]:


x[:, :]


# Selects all rows and all columns starting from the second column of the tensor x in one line.

# In[38]:


x[:, 1:]


# Selects all rows and all columns starting from the third column of the tensor x.

# In[39]:


x[:, 2:]


# selects a sub-tensor from the tensor x, excluding the first row and including columns 1 and 2.

# In[41]:


x[1:, 1:3]


# torch.split is used to split a tensor into multiple chunks along a specified dimension.

# In[49]:


x_rows = torch.split(x, split_size_or_sections=1, dim=0)
print(x_rows)


# In[42]:


x_cols = torch.split(x, split_size_or_sections=2, dim=1)
print(x_cols)


# In[43]:


torch.chunk(x, chunks=2, dim=1)


# In[44]:


torch.masked_select(x_cols[0], torch.BoolTensor([[0,1],[1,0],[0,1]]).bool())


# ### Tensor Merging :

# Tensor merging refers to combining multiple tensors into a single tensor, typically along a specified dimension or using a concatenation operation.

# torch.cat concatenates a sequence of tensors along a specified dimension

# In[50]:


torch.cat(x_rows, dim = 0)
print(x)


# In[51]:


torch.cat(x_rows, dim=1)


# torch.stack creates a new tensor by stacking a sequence of tensors along a new dimension in one line.

# In[52]:


x_new = torch.stack(x_cols, dim=0)
print(x_new)


# ### Tensor Reshaping :

# In[53]:


x = torch.Tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(x)


# torch.view views the tensor in specified rows an columns

# In[54]:


x.view(4, 3)


# In[55]:


x.view(-1, 2)


# In[56]:


x.view(2, 1, -1)


# In[57]:


x.view(2, 1, -1).squeeze()


# In[58]:


x.view(2, 1, -1).squeeze().unsqueeze(dim=1)


# ### Tensor Calculation :

# Tensor calculations involve performing various mathematical operations, such as addition, subtraction, multiplication, division, and more, on tensors. These operations can be element-wise or involve operations between entire tensors, often used in mathematical modeling, machine learning, and scientific computing.

# In[59]:


x = torch.Tensor([[1,2,3],[4,5,6]])
y = torch.Tensor([[1,1,1],[2,2,2]])
print("x :", x)
print("y :", y)


# In[63]:


z = torch.add(x, y)
print(x)
print('z :', z )
print('x + y :', x+y)


# In[64]:


print(x - 2)


# In[65]:


print(x*2)


# In[66]:


print(2*x)


# In[67]:


z = torch.mul(x, y)
print("z :", z)
print("x*y :", x*y)


# In[68]:


z = torch.pow(x, 2)
print("z :", z)
print("x**2 :", x**2)


# In[69]:


z = torch.log(x)
print("z :", z)
print("x.log() :", x.log())


# In[70]:


z = torch.sqrt(x)
print("z :", z)
print("x.sqrt() :", x.sqrt())


# In[71]:


z = x % 2
print(z)


# In[72]:


z = torch.abs(x)
print("z :", z)
print("x.abs() :", x.abs())


# ### Tensor Casting : 

# Tensor casting refers to changing the data type of a tensor from one numerical type to another, such as converting from integers to floating-point numbers or vice versa. This can be important for ensuring compatibility in mathematical operations and precision requirements.

# In[73]:


x = torch.Tensor([[1,2,3],[4,5,6]])
y = torch.Tensor([[1,1,1],[2,2,2]])
print("x :", x)
print("y :", y)


# In[74]:


x.type(torch.DoubleTensor)


# In[75]:


x.double()


# In[76]:


x.type(torch.IntTensor)


# In[78]:


x.int()


# ### Tensor Statics :

# 
# Tensor statistics involve calculating various numerical properties of tensors, such as mean, standard deviation, minimum, and maximum values, which are essential for understanding the data distribution and performing data analysis.

# In[79]:


x = torch.Tensor([[-1,2,-3],[4,-5,6]])
print(x)


# In[82]:


print('Sum :', x.sum())
print('Min :', x.min())
print('Max :', x.max())
print('Var :', x.var())


# In[83]:


x.sum()


# In[84]:


x.sum().size()


# In[85]:


print('Sum Item:', x.sum().item())
print('Min Item:', x.min().item())
print('Max Item:', x.max().item())
print('Var Item :', x.var().item())


# In[86]:


value, index = x.max(dim=0)
value, index


# In[87]:


value, index = x.max(dim=1)
value, index


# In[88]:


value, index = x.sort(dim=1)
value


# ### Like Function :

# In[89]:


x = torch.Tensor([[-1,2,-3],[4,-5,6]])
print(x)


# 
# torch.zeros_like creates a tensor of zeros with the same shape as the input tensor 

# In[90]:


y = torch.zeros_like(x)
print(y)


# 
# torch.ones_like creates a tensor of ones with the same shape as the input tensor 

# In[91]:


y = torch.ones_like(x)
print(y)


# In[92]:


y = torch.rand_like(x)
print(y)


# ### Tensor Calculation As Matrix  :

# 
# Performing tensor calculations as matrices involves using mathematical operations on tensors, often in a matrix-like manner, to solve linear equations, transform data, or perform operations like matrix multiplication, inversion, and decomposition.

# In[96]:


x = torch.Tensor([[1,1],[2,2]])
y = torch.Tensor([[1,2],[3,4]])
print("x :", x)
print("y :", y)


# 
# torch.mm computes the matrix multiplication between two tensors in one line.

# In[97]:


torch.mm(x, y)


# In[98]:


z = torch.Tensor([1, 2])
torch.mv(x, z)


# `torch.dot` calculates the dot product between two 1-dimensional tensors in one line.

# In[99]:


torch.dot(z, z)


# In[100]:


x.t()


# In[101]:


y.inverse()


# In[102]:


x1 = torch.FloatTensor(3,3,2)
x2 = torch.FloatTensor(3,2,3)

torch.bmm(x1,x2).size()


# In[103]:


x1


# In[104]:


x2


# 
# x.transpose returns the transpose of a tensor x, swapping its dimensions in one line.

# In[105]:


x1.transpose(0,2)


# torch.linalg.eig computes the eigenvalues and eigenvectors of a square matrix in one line.

# In[108]:


eigenvalue, eigenvector = torch.linalg.eig(x)
print("eigenvalue :", eigenvalue)
print("eigenvector :", eigenvector)


# torch.qr computes the QR decomposition of a matrix, factorizing it into an orthogonal and an upper triangular matrix in one line.

# In[109]:


Q,R = torch.qr(x)
print("Q :", Q)
print("R :", R)


# torch.svd performs Singular Value Decomposition on a tensor, factorizing it into three tensors in one line.

# In[110]:


S,V,D = torch.svd(x)
print("S :", S)
print("V :", V)
print("D :", D)


# In[ ]:




