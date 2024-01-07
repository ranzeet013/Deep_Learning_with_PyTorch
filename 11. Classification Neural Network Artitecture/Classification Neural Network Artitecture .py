#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
from torchinfo import summary

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles


# In[3]:


n_samples = 1000
x, y = make_circles(n_samples, 
                    noise = 0.03, 
                    random_state = 42)

len(x), len(y)


# In[4]:


x[:5], y[:5]


# In[8]:


circles = pd.DataFrame({"X1": x[:, 0], 
                        "X2": x[:, 1], 
                        "label": y})

circles.head()


# In[13]:


plt.figure(figsize = (5, 4))
plt.scatter(x = x[:, 0], 
            y = x[:, 1], 
            c = y, 
            cmap = plt.cm.RdYlBu)


# In[15]:


x = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

x[:5], y[:5]


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(x, y,                       #splitting the dataset into training and testing set
                                                    test_size = 0.2,
                                                    random_state = 42)

len(x_train), len(x_test), len(y_train), len(y_test)


# In[27]:


#model with 2 layers 

class modelCircle(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features = 2, out_features = 5)       #take 2 features and upscales to 5 features
        self.layer_2 = nn.Linear(in_features = 5, out_features = 1)       #take 5 features from above layer and output 1 features  
        
    def forward(self, x):
        return self.layer_2(self.layer_1(x))         #output of layer 1 goes to layer 2
    
model = modelCircle()
model, summary(model)


# In[32]:


#replicating the same model using nn.Sequential

model1 = nn.Sequential(
    nn.Linear(in_features = 2, out_features = 5),
    nn.Linear(in_features = 5, out_features = 1)
)

model1, 
summary(model1), 


# In[33]:


model1.state_dict()


# In[37]:


predictions = model1(x_test)

print(f"Length of predictions: {len(predictions)}, Shape: {predictions.shape}")
print(f"\nFirst 10 Predictions: \n{predictions[:10]}")
print(f"\nFirst 10 Labels: \n{y_test[:10]}")


# In[42]:


model1.eval()
with torch.inference_mode():
    

    y_logits = model1(x_test)[:5]
y_logits


# In[43]:


y_test[:5]


# In[44]:


y_pred_probs = torch.sigmoid(y_logits)
y_pred_probs


# In[45]:


torch.round(y_pred_probs)


# In[38]:


loss_fn = nn.BCEWithLogitsLoss()  #builtin sigmoid activaion function 

optimizer = torch.optim.SGD(params = model1.parameters(), 
                            lr = 0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc 


# In[56]:


torch.manual_seed(42)

epochs = 100

x_train, y_train = x_train, y_train
x_test, y_test = x_test, y_test

for epoch in range(epochs):
    model1.train()
    
    y_logits = model1(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))  #logits -> pred probs -> pred labels
    
    loss = loss_fn(y_logits,     # raw logit as input, nn.BCEWithLoss
                   y_train)
    
    acc = accuracy_fn(y_true = y_train,
                      y_pred = y_pred)
    
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()    #(gradient decent) optimizer_step
    
    #testing
    model1.eval()
    with torch.inference_mode():
        test_logits = model1(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        # calculate accuracy / loss
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true = y_test, 
                               y_pred = test_pred)
        
        
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.2f}%")


# In[63]:


#model with 3 layers 

class modelCircle(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features = 2, out_features = 10)       #take 2 features and upscales to 10 features
        self.layer_2 = nn.Linear(in_features = 10, out_features = 10)      #take 10 features from above layer_2 and output 10 features  
        self.layer_3 = nn.Linear(in_features = 10, out_features = 1)       #take 10 features from above layer_3 and output 1 feature
        
    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))         #output of layer 1 goes to layer 2
    
model = modelCircle()
model, summary(model)


# In[64]:


model.state_dict()


# In[65]:


loss_fn = nn.BCEWithLogitsLoss()  #builtin sigmoid activaion function 

optimizer = torch.optim.SGD(params = model.parameters(), 
                            lr = 0.1)


# In[67]:


torch.manual_seed(42)

epochs = 1000

x_train, y_train = x_train, y_train
x_test, y_test = x_test, y_test

for epoch in range(epochs):
    model.train()
    
    y_logits = model(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))  #logits -> pred probs -> pred labels
    
    loss = loss_fn(y_logits,     # raw logit as input, nn.BCEWithLoss
                   y_train)
    
    acc = accuracy_fn(y_true = y_train,
                      y_pred = y_pred)
    
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()    #(gradient decent) optimizer_step
    
    #testing
    model.eval()
    with torch.inference_mode():
        test_logits = model(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        # calculate accuracy / loss
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true = y_test, 
                               y_pred = test_pred)
        
        
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.2f}%")


# In[ ]:




