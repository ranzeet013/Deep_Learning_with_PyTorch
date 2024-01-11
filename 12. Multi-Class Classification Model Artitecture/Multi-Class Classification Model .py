#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

import torch 
from torch import nn
from torchinfo import summary

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split



# In[2]:


NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

#hyperparameter for data creation
x_blob, y_blob = make_blobs(n_samples = 1000, 
                            n_features = NUM_FEATURES, 
                            centers = NUM_CLASSES, 
                            cluster_std = 1.5, 
                            random_state = RANDOM_SEED)

#turning data into tensors 
x_blob = torch.from_numpy(x_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)


#splitting the dataset into training set and testing set 
x_blob_train, x_blob_test, y_blob_train, y_blob_test = train_test_split(x_blob, 
                                                                        y_blob, 
                                                                        test_size = 0.2, 
                                                                        random_state = RANDOM_SEED)

x_blob_train.shape, x_blob_test.shape, y_blob_train.shape, y_blob_test.shape   #shapes of splits

#plotting the dataset 
plt.figure(figsize = (5, 3))       
plt.scatter(x_blob[:, 0], 
            x_blob[:, 1], 
            c = y_blob, 
            cmap = plt.cm.RdYlBu)


# In[3]:


#multi-class classification model

class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units = 8):
        """Initializes multi-class classification model.
        
        Args:
           input_features = number of input features to the model 
           output_features = numbrt of output features 
           hidden units = number of hidden units between the layers , default is set to 8
        
        Returns :
        
        Example :
        
        """
        
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features = input_features, out_features = hidden_units), 
            nn.ReLU(), 
            nn.Linear(in_features = hidden_units, out_features = hidden_units), 
            nn.ReLU(), 
            nn.Linear(in_features = hidden_units, out_features = output_features)
        )
        
    def forward(self, x):
        return self.linear_layer_stack(x)
        
model = BlobModel(input_features = 2, 
                  output_features = 4, 
                  hidden_units = 8)

model, 

summary(model)


# In[4]:


torch.unique(y_blob_train)


# In[5]:


#loss function and optimizer

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params = model.parameters(), 
                            lr = 0.1)


# In[6]:


model.eval()
with torch.inference_mode():
    y_logits = model(x_blob_test)
    
y_logits[:10]                                      #logits


# In[7]:


y_pred_probs = torch.softmax(y_logits, dim = 1)                  #logits -> prediction_probabilities
print(y_logits[:5])
print(y_pred_probs[:5])


# In[8]:


y_preds = torch.argmax(y_pred_probs, dim = 1)                    #prediction_probabilites -> predictions_labels
y_preds


# In[13]:


def accuracy_fn(y_true, y_pred):
    correct = (y_pred == y_true).sum().item()
    total = len(y_true)
    accuracy = correct / total
    return accuracy


# In[14]:


torch.manual_seed(42)

epochs = 100

x_blob_train, y_blob_train = x_blob_train, y_blob_train
x_blob_test, y_blob_test = x_blob_test, y_blob_test

for epoch in range(epochs):
    model.train()
    
    y_logits = model(x_blob_train)
    
    y_pred = torch.softmax(y_logits, dim = 1).argmax(dim = 1)
    
    #loss and accuracy
    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true = y_blob_train,
                      y_pred = y_pred)
    
    #optimizer and loss function 
    optimizer.zero_grad()
    
    loss.backward()
    optimizer.step()
    
    #testing
    model.eval()
    with torch.inference_mode():
        test_logits = model(x_blob_test)
        test_preds = torch.softmax(test_logits, dim = 1).argmax(dim = 1)
        
        #test accuracy / loss
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true = y_blob_test,
                               y_pred = test_preds)
        
        
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss: .4f}, Acc: {acc:.2f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")


# In[19]:


def plot_decision_boundary(model, X, y):
    assert X.shape[1] == 2
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()])).detach().numpy()
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha = 0.3)
    plt.scatter(X[:, 0], X[:, 1], c = y, 
                edgecolors = 'k',
                marker = 'o',
                s = 80,
                linewidth = 1)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')

plt.figure(figsize=(5, 4))
plt.subplot(1, 2, 1)
plt.title('Train')
plot_decision_boundary(model, x_blob_train, y_blob_train)

plt.subplot(1, 2, 2)
plt.title('Test')
plot_decision_boundary(model, x_blob_test, y_blob_test)

plt.show()


# In[ ]:




