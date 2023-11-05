#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn


# 
# 1. `BasicBlock` is a PyTorch neural network module representing a basic building block for deep learning models.
# 
# 2. `multiplier=1` is a class variable specifying the scaling factor for the number of output channels in the convolutional layers.
# 
# 3. The `__init__` method initializes the `BasicBlock` module, taking input channel count, desired output channel count, and an optional stride as parameters.
# 
# 4. `conv_layer1` is the first convolutional layer with 3x3 kernel, batch normalization, and ReLU activation.
# 
# 5. `conv_layer2` is the second convolutional layer with similar properties as `conv_layer1`.
# 
# 6. `res_connnection` is a residual connection initialized as an empty sequential container.
# 
# 7. Conditional check in `__init__` determines if the residual connection is needed based on stride and input channel count.
# 
# 8. The `forward` method computes the forward pass of the module by applying convolutions, batch normalization, and residual connections with ReLU activation, and returns the output.

# In[4]:


class BasicBlock(nn.Module):
    multiplier=1
    def __init__(self, input_num_planes, num_planes, strd = 1):
        super(BasicBlock, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels = input_num_planes, out_channels = num_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.batch_norm1 = nn.BatchNorm2d(num_planes)
        self.conv_layer2 = nn.Conv2d(in_channels = num_planes, out_channels = num_planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.batch_norm2 = nn.BatchNorm2d(num_planes)
 
        self.res_connnection = nn.Sequential()
        if strd > 1 or input_num_planes != self.multiplier*num_planes:
            self.res_connnection = nn.Sequential(
                nn.Conv2d(in_channels = input_num_planes, out_channels = self.multiplier*num_planes, kernel_size = 1, stride = strd, bias = False),
                nn.BatchNorm2d(self.multiplier*num_planes)
            )
    def forward(self, inp):
        op = F.relu(self.batch_norm1(self.conv_layer1(inp)))
        op = self.batch_norm2(self.conv_layer2(op))
        op += self.res_connnection(inp)
        op = F.relu(op)
        return op


# In[ ]:




