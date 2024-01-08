#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
import torch.utils.data as Data

torch.manual_seed(6)


# ### Craeting Dataset :

# Creating two tensors, x and y, each containing 10 evenly spaced values within specified ranges and  linear sequences from 1 to 10 for x, and from 10 to 1 for y, representing data points then concatenates tensors x and y by reshaping them into column vectors and stacking them side by side.

# In[8]:


x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch.cat((x.view(len(x),-1),y.view(len(y),-1)),1)


# In[11]:


dataset = Data.TensorDataset(x, y)


# In[13]:


dataset


# ### Training :

# Training a model using a DataLoader for efficient data handling which iterates through three epochs, processing data in batches of five. Each epoch's progress and batch data are printed, aiding in training monitoring.

# In[14]:


batch_size = 5

loader = Data.DataLoader(dataset = dataset, 
                         batch_size = batch_size, 
                         shuffle = True, 
                         num_workers = 1)


# In[15]:


for epoch in range (3):
    for step, (batch_x, batch_y) in enumerate(loader):
        print('Epoch :', epoch, ' | Step :', step, ' | batch_x :', batch_x.numpy(), ' | batch_y :', batch_y.numpy())


# Training a model using a DataLoader for efficient data handling which iterates through three epochs, processing data in batches of ten. Each epoch's progress and batch data are printed, aiding in training monitoring.

# In[18]:


batch_size = 10

loader = Data.DataLoader(dataset = dataset, 
                         batch_size = batch_size, 
                         shuffle = True, 
                         num_workers = 1)

for epoch in range (3):
    for steps, (batch_x, batch_y) in enumerate (loader):
        print('Epoch :', epoch, '| steps :', step, '|batch_x :', batch_x.numpy(), '|batch_y :', batch_y.numpy())


# In[19]:


import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms


# In[20]:


train_dataset = dsets.MNIST(root = './data', 
                            train = True, 
                            transform = transforms.ToTensor(), 
                            download = True)

image, label = train_dataset[0]
print(image.size())
print(label)


# ### Batch Training with Image Dataset ( MNIST ) :

# ### Loading Dataset :

# DataLoader named train_dataloader to manage a training dataset. It configures the DataLoader to process batches of 100 samples each, shuffle the data before batching, and utilize 2 worker processes for parallel data loading.

# In[24]:


train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                               batch_size = 100, 
                                               shuffle = True, 
                                               num_workers = 2)


# In[27]:


for images, labels in train_dataloader:
    pass


# In[28]:


data_iter = iter(train_dataloader)
batch = next(data_iter)


# The ResNet-18 model is loaded with pre-trained weights which modifies the fully connected (fc) layer to accommodate a new output size of 100. Random input images (10 samples) are created, passed through the modified ResNet, and the resulting output size is printed the setting up the gradient computation of the pre-trained layers to false to retain the pre-trained knowledge during training.

# In[30]:


resnet = torchvision.models.resnet18(pretrained = True)                    #pretrained model

for params in resnet.parameters():
    params.requires_grad = False
    
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 100)

images = torch.autograd.Variable(torch.randn(10, 3, 256, 256))
outputs = resnet(images)
print (outputs.size()) 


# In[ ]:




