#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn


# 
# 
# 1. `self.batch_norm1 = nn.BatchNorm2d(input_num_planes)`: Initializes a 2D batch normalization layer for the input feature maps to stabilize training.
# 2. `self.conv_layer1 = nn.Conv2d(in_channels=input_num_planes, out_channels=4*rate_inc, kernel_size=1, bias=False)`: Creates a 1x1 convolutional layer to increase channel dimension by a factor of 4, without bias.
# 3. `self.batch_norm2 = nn.BatchNorm2d(4*rate_inc)`: Defines another batch normalization layer for the output of the first convolution.
# 4. `self.conv_layer2 = nn.Conv2d(in_channels=4*rate_inc, out_channels=rate_inc, kernel_size=3, padding=1, bias=False)`: Adds a 3x3 convolutional layer with padding to reduce the channel dimension to `rate_inc`.
# 5. `op = self.conv_layer1(F.relu(self.batch_norm1(inp)))`: Applies batch normalization, ReLU activation, and the first convolution to the input.
# 6. `op = self.conv_layer2(F.relu(self.batch_norm2(op)))`: Performs batch normalization, ReLU activation, and the second convolution on the previous output.
# 7. `op = torch.cat([op, inp], 1)`: Concatenates the output of the second convolution with the original input along the channel dimension.
# 8. `return op`: Returns the final output after the transformations in the `forward` method.

# In[2]:


class DenseBlock(nn.Module):
    def __init__(self, input_num_planes, rate_inc):
        super(DenseBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(input_num_planes)
        self.conv_layer1 = nn.Conv2d(in_channels=input_num_planes, out_channels=4*rate_inc, kernel_size=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(4*rate_inc)
        self.conv_layer2 = nn.Conv2d(in_channels=4*rate_inc, out_channels=rate_inc, kernel_size=3, padding=1, bias=False)
    def forward(self, inp):
        op = self.conv_layer1(F.relu(self.batch_norm1(inp)))
        op = self.conv_layer2(F.relu(self.batch_norm2(op)))
        op = torch.cat([op,inp], 1)
        return op


# PyTorch module `TransBlock`, which is used for transitioning between different stages or levels of feature maps in a convolutional neural network (CNN).
# 
# 1. `self.batch_norm = nn.BatchNorm2d(input_num_planes)`: Initializes a 2D batch normalization layer for the input feature maps to stabilize training.
# 
# 2. `self.conv_layer = nn.Conv2d(in_channels=input_num_planes, out_channels=output_num_planes, kernel_size=1, bias=False)`: Creates a 1x1 convolutional layer that transforms the input feature maps from `input_num_planes` channels to `output_num_planes` channels, without bias.
# 
# 3. `op = self.conv_layer(F.relu(self.batch_norm(inp))`: Applies batch normalization, ReLU activation, and the 1x1 convolution to the input feature maps, transforming them to the desired number of output channels.
# 
# 4. `op = F.avg_pool2d(op, 2)`: Performs average pooling with a 2x2 window to downsample the spatial dimensions of the feature maps, reducing their size. This operation is often used to transition from one stage of a CNN to another while reducing spatial resolution.

# In[3]:


class TransBlock(nn.Module):
    def __init__(self, input_num_planes, output_num_planes):
        super(TransBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(input_num_planes)
        self.conv_layer = nn.Conv2d(in_channels=input_num_planes, out_channels=output_num_planes, kernel_size=1, bias=False)
    def forward(self, inp):
        op = self.conv_layer(F.relu(self.batch_norm(inp)))
        op = F.avg_pool2d(op, 2)
        return op


# In[ ]:




